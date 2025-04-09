import os
import socket
from mpi4py import MPI
from accelerate import Accelerator
import torch

def next_free_port( port=1024, max_port=65535 ):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')
    
def init_distributed_env(accelerate_ranks=None, accelerate_kwargs=None):
    """
    - Initializes MPI.
    - Optionally configures environment variables so that the ranks
      in `accelerate_ranks` can form a smaller "Accelerate world."
    - Returns:
        comm, world_rank, world_size, accelerator
    """
    if accelerate_kwargs is None:
        accelerate_kwargs = {}
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if accelerate_ranks is None:
        # Default: only rank 0 will use Accelerate
        accelerate_ranks = [0]

    # Broadcast a common MASTER_ADDR from rank 0
    if world_rank == 0:
        assert world_rank in accelerate_ranks, "Rank 0 must be in accelerate ranks"
        master_addr = socket.gethostname()
        # TODO: broadcast a port+addr for the case with multiple accelerate_ranks?
    elif world_rank not in accelerate_ranks:
        # i think this helps prevent accelerate from joining all processes
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["RANK"] = str(0)
        os.environ["LOCAL_RANK"] = str(0)
        os.environ["MASTER_ADDR"] = socket.gethostname()
        os.environ["MASTER_PORT"] = f"{next_free_port(29500+world_rank*10)}"  # pick an unused port

    accelerator = None
    if world_rank in accelerate_ranks:
        # TODO: modify/test this for more than just 1 accelerator rank 
        local_accel_rank = accelerate_ranks.index(world_rank)

        os.environ["WORLD_SIZE"] = str(len(accelerate_ranks))
        os.environ["RANK"] = str(local_accel_rank)
        os.environ["LOCAL_RANK"] = str(local_accel_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = f"{next_free_port(29500+world_rank*10)}"  # pick an unused port

        # Create the Accelerator only on these sub-world ranks
        accelerator = Accelerator(**accelerate_kwargs)

    return comm, world_rank, world_size, accelerator


def broadcast_weights(model, comm: MPI.Comm, root_mpi_rank: int):
    """
    Broadcast all of `model`'s parameters from `root_mpi_rank`
    to every other MPI rank. If you're running on GPU,
    we must temporarily copy parameters to CPU for the broadcast.

    TODO: make this faster for larger-scale models (e.g. >=7B params)
    """
    world_rank = comm.Get_rank()
    for name, param in model.named_parameters():
        # conversion from bfloat needed for Bcast, it's a no-op if float32 already
        param_cpu = param.data.to(torch.float32).cpu().numpy() 
        # Broadcast in-place from root
        comm.Bcast(param_cpu, root=root_mpi_rank)
        # Non-root ranks copy data back into model param
        if world_rank != root_mpi_rank:
            # Convert back to original dtype and device
            param.data = torch.from_numpy(param_cpu).to(
                dtype=param.data.dtype,
                device=param.data.device
            )