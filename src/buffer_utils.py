import math
import random
import time
from mpi4py import MPI
import numpy as np
import torch


class CommentBuffer:
    """
    Maintains a list of items up to max_capacity, provides
    deduplication, merges, and sampling (trainer uses sampling).
    """

    def __init__(self, cid, max_capacity=240, sample_var='score', inv_temp=1):
        self.cid = cid
        self.items = [] # Each item: { "response":..., "advantage":..., "policys_trainer_iteration":... }
        self.max_capacity = max_capacity
        self.last_synced_trainer_iteration = -1
        self.n_online = 0
        self.sample_var = sample_var
        self.inv_temp = inv_temp

    def add_new_items(self, new_items, trainer_iteration):
        """
        Merge new_items with self.items, deduplicate, keep only newest duplicates, keep last max_items
        Used by workers after generating new local items.
        """
        if trainer_iteration>self.last_synced_trainer_iteration:
            self.n_online = 0
            self.last_synced_trainer_iteration = trainer_iteration
            
        self._deduplicate_and_keep_newest(self.items, new_items, trainer_iteration)
        self._compute_reward_based_probs()

    def _deduplicate_and_keep_newest(self, old_items, new_items, trainer_iteration):
        """
        Combine old_items + new_items, deduplicate by token-hash so that
        for each token-hash, we keep only the item with the greatest 'last_synced_trainer_iteration'.
        Then sort by ascending last_synced_trainer_iteration, keep the newest max_capacity.
        """
        temp = {}
        for it in new_items:
            t = tuple(it["response"])
            if t not in temp:
                temp[t] = it 
                self.n_online += 1
        for it in old_items:
            t = tuple(it["response"])
            if t not in temp:
                temp[t] = it 
        deduped = list(temp.values())
        deduped.sort(key=lambda x: x["policys_trainer_iteration"])
        self.items = deduped[-self.max_capacity:]
            
    def _compute_reward_based_probs(self):
        """
        Compute the softmax distribution over 'reward' for sampling.
        """
        priorities = [item[self.sample_var]*self.inv_temp for item in self.items]
        priorities = np.array(priorities)
        priorities = priorities - np.max(priorities)
        priorities = np.exp(priorities)
        self.prob = priorities / np.sum(priorities)

    def get_batch(self, num_samples, online=True):
        replace = False
        if len(self.items) < num_samples:
            print(f'Sampling with replacement because CID {self.cid} has only {len(self.items)} samples',
                  flush=True)
            # this happens when we didn't add enough non-duplicative samples
            replace=True
        
        if not online: #do reward-based sampling
            idx = np.random.choice(len(self.items), num_samples, p=self.prob, replace=replace)
        else:
            idx = np.random.choice(
                range(len(self.items)-self.n_online,
                      len(self.items)
                ),
                num_samples,
                replace=self.n_online<num_samples
            )
            
        batch = {
            'responses': [],
            'advantages': [],
            'scores': [],
            'logprobs': [],
            'ref_logprobs': [],
            'sequence_lengths': []
        }
        for k in batch:
            batch[k] = [self.items[i][k[:-1]] for i in idx] #remove the 's' from the key
            
        return batch

class CommentBuffersManager:
    """
    Holds a dictionary of { cid -> CommentBuffer }.
    - Trainer rank: has a CommentBuffer for every cid in assigned_comment_ids.
    - Worker ranks: have a CommentBuffer for mutually exclusive cid subsets.
    """

    def __init__(
        self, 
        assigned_comment_ids, 
        rloo_k = 1, 
        online_prob = 1,
        max_capacity_per_query = 100,
        sample_var = 'score',
        inv_temp = 1
    ):

        self.assigned_comment_ids = assigned_comment_ids
        self.comment_buffers = {}
        for cid in assigned_comment_ids:
            self.comment_buffers[cid] = CommentBuffer(
                cid,
                max_capacity=max_capacity_per_query,
                sample_var=sample_var,
                inv_temp=inv_temp
            )

        self.rloo_k = rloo_k
        self.online_prob = online_prob
        self.last_synced_trainer_iteration = -1
        self.offline_cids = set() # equal to self.assigned_comment_ids only if we fully init buffer
        self.seen_comment_ids = set() # instead using this to avoid fully initializing 
        self.online_cids = set()

    def overwrite_cid_buffer(self, cid, new_data, training_iteration):
        if training_iteration>self.last_synced_trainer_iteration:
            # this is the first update in a new sync step
            self.last_synced_trainer_iteration = training_iteration
            self.online_cids = set()
            self.removed_online_cids = set()
        self.comment_buffers[cid] = new_data
        self.online_cids.add(cid)
        self.seen_comment_ids.add(cid) # need this to avoid full initialization

    def sample_cid(self, online=True):
        if online:
            if len(self.online_cids)==0:
                print('\nOut of online samples. Setting online CIDs to recently used online CIDs',flush=True)
                self.online_cids = set(self.removed_online_cids)
            cid = np.random.choice(list(self.online_cids))
            self.online_cids.remove(cid)
            self.removed_online_cids.add(cid)
            return cid
        if len(self.offline_cids) == 0: # could reset to self.assigned_comment_ids if we fully init buffer
            print('\nOut of offline samples. Setting offline CIDs to all seen CIDs',flush=True)
            self.offline_cids = set(self.seen_comment_ids) 
        cid = np.random.choice(list(self.offline_cids))
        self.offline_cids.remove(cid)
        return cid

    def get_batch(self, num_samples):
        num_cids_sampled = num_samples // self.rloo_k
        if num_samples != num_cids_sampled * self.rloo_k:
            raise ValueError("`num_samples` must be a multiple of `rloo_k`, inexact division:"+
                             f"{num_samples} / {self.rloo_k} = {num_samples / self.rloo_k}")
        
        batch = {
            'responses': [],
            'advantages': [],
            'scores': [],
            'logprobs': [],
            'ref_logprobs': [],
            'sequence_lengths': []
        }
        
        cids_sampled = []
        for _ in range(num_cids_sampled):
            online = torch.rand(1) < self.online_prob
            cid = self.sample_cid(online=online)
            cids_sampled.append(cid)
            
            mb = self.comment_buffers[cid].get_batch(
                self.rloo_k, 
                online=online
            )

            for k in batch:
                batch[k]+=mb[k]
        
        for k in batch:
            batch[k] = torch.from_numpy(np.stack(batch[k]))
        return batch, cids_sampled