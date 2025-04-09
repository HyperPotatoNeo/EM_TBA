import os
import math
import multiprocessing
import random

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def example_prepare_dataset(dataset, tokenizer, max_length=512):
    """
    Pre-tokenize the dataset before training; only collate during training.
    """
    def tokenize(element):
        input_ids = tokenizer(
            element["query"],
            padding=False,  # do not pad here; collate_fn will do it later
        )["input_ids"]
        return {
            "input_ids": input_ids,
            "lengths": [len(ids) for ids in input_ids]
        }

    # Tokenize in parallel
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
    )

    # Filter out anything over max_length
    dataset = dataset.filter(lambda x: x["lengths"] <= max_length)
    return dataset


def build_example_dataloader(dataset, tokenizer, batch_size=16):
    """
    Build a reference dataloader over the entire dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
        drop_last=False
    )
    
def split_dataset_indices(total_size, K):
    """
    Split global indices [0..N-1] among K processes in a round-robin fashion.
    i.e. process 0 gets indices [0, K, 2K, ...],
         process 1 gets indices [1, K+1, 2K+1, ...], etc.
    """
    all_indices = list(range(total_size))
    indices_for_processes = []
    for i in range(K):
        cids_for_this_proc = all_indices[i::K]
        indices_for_processes.append(cids_for_this_proc)
    return indices_for_processes


class SubsetByCidDataset(Dataset):
    """
    Wraps an original dataset but restricts it to the subset of cids given.

    Also allows retrieval by global cid via a mapping cid -> local index.
    """
    def __init__(self, original_dataset, cids):
        self.original_dataset = original_dataset
        self.cids = list(cids)
        # Map from cid (global index) -> local index in [0..len(cids)-1]
        self.cid2local_idx = {cid: i for i, cid in enumerate(self.cids)}

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, local_idx):
        """
        local_idx in [0..len(cids)-1].
        We map local_idx -> global_idx -> fetch from original dataset.
        """
        global_idx = self.cids[local_idx]
        item = self.original_dataset[global_idx]
        item['cid'] = global_idx
        return item

class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterator)

############################################################################
# Tests
############################################################################

if __name__ == "__main__":

    # Detect rank and K from SLURM environment (fallback defaults if not under SLURM).
    rank = int(os.environ.get("SLURM_PROCID", 0))
    K = int(os.environ.get("SLURM_NTASKS", 1))

    dataset_name = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    raw_datasets = load_dataset(dataset_name)

    # same splits used in your snippet
    train_dataset_raw = raw_datasets["train"]
    eval_dataset_raw = raw_datasets["validation"]

    tokenizer = AutoTokenizer.from_pretrained(
        "mnoukhov/pythia410m-sft-tldr",
        padding_side="left",
        trust_remote_code=True,
    )

    # Preprocess training set
    train_dataset_prepared = example_prepare_dataset(train_dataset_raw, tokenizer, max_length=512)

    # The total size after filtering
    N = len(train_dataset_prepared)
    if rank == 0:
        print(f"[Rank 0] Filtered train_dataset size = {N}")

    # Build a reference dataloader (over entire dataset)
    reference_batch_size = 16 * K
    reference_dataloader = build_example_dataloader(
        train_dataset_prepared,
        tokenizer,
        batch_size=reference_batch_size
    )
    # Grab the first batch from the reference dataloader
    reference_first_batch = next(iter(reference_dataloader))
    data_collator = DataCollatorWithPadding(tokenizer)
    recreated = data_collator([train_dataset_prepared[i] for i in range(64)])['input_ids']
    test_passed = all([torch.equal(reference_first_batch["input_ids"][i], recreated[i]) for i in range(16)])
    print(f'data collator test: {test_passed}')
    assert test_passed
    
    # Create the local subset dataset for this rank
    all_cids = split_dataset_indices(N, K)
    my_cids = all_cids[rank]
    local_dataset = SubsetByCidDataset(train_dataset_prepared, my_cids)

    local_dataloader_batch_size = 16
    local_dataloader = build_example_dataloader(
        local_dataset,
        tokenizer,
        batch_size=local_dataloader_batch_size
    )

    # Print a few local batches
    print(f"\n[Rank {rank}] Local dataloader (subset) - first 3 batches")
    local_first_batch = None
    for i, batch in enumerate(local_dataloader):
        if i == 0:
            local_first_batch = batch
        print(f"  [Rank {rank}] Batch {i} => keys: {list(batch.keys())}, context lengths: {[len(x) for x in batch['input_ids']]}")
        if i >= 2:
            break

    print(f'local dataset has cid {local_first_batch["cid"]} on first item')
    ############################################################################
    # Compare a few items in local_first_batch vs. the reference_first_batch
    #
    # WARNING: This comparison only works if my_cids[i] < reference_batch_size,
    # because we're directly indexing reference_first_batch with [cid].
    # If your cids exceed the reference batch size, you'll get an IndexError.
    ############################################################################

    print(f"\n[Rank {rank}] Checking item equivalence for up to 4/{len(local_dataloader.dataset)} local items with CIDs {my_cids[:4]}...")
    print(f"\n[Rank {rank}] Has {len(local_dataloader)} batches...")
    for i, cid in enumerate(my_cids[:4]):
        if cid >= reference_batch_size:
            print(f"  [Rank {rank}] Skipping cid={cid}, because reference_first_batch has size {reference_batch_size}.")
            continue

        # local_first_batch[i] => the i-th sample in the local batch
        # reference_first_batch[cid] => the cid-th sample in the reference batch
        # These might match only if the underlying item is truly the same row in the dataset.
        local_item = {
            'input_ids': local_first_batch['input_ids'][i],
            'lengths': local_first_batch['lengths'][i],
        }
        ref_item = {
            'input_ids': reference_first_batch['input_ids'][cid],
            'lengths': reference_first_batch['lengths'][cid],
        }

        same_length = (local_item["lengths"] == ref_item["lengths"])
        n_extra_padding_tokens = max(reference_first_batch['lengths']) - max(local_first_batch['lengths'])
        same_input_ids = torch.equal(local_item["input_ids"], ref_item["input_ids"][n_extra_padding_tokens:])

        if same_input_ids and same_length:
            print(f"  [Rank {rank}] i={i}, cid={cid} => MATCH [actual length={local_item['lengths'], ref_item['lengths']}; content={local_item['input_ids'][-20:-15], ref_item['input_ids'][-20:-15]}]")
        else:
            print(f"  [Rank {rank}] i={i}, cid={cid} => MISMATCH:")
            print(f"     local_item['input_ids'] = {local_item['input_ids']}")
            print(f"     ref_item['input_ids']   = {ref_item['input_ids']}")
            print(f"     local_item['lengths']   = {local_item['lengths']}")
            print(f"     ref_item['lengths']     = {ref_item['lengths']}")

    print(f"\n[Rank {rank}] All done!\n")
