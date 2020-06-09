import os
from math import ceil
import natsort


def chunkbasename(filename):
    return f"{filename}.chunk."


def is_chunkname(chunkname, filename):
    return chunkname.rstrip("0123456789") == chunkbasename(filename)


def get_chunkname(filename, i_chunk):
    return f"{chunkbasename(filename)}{i_chunk}"


def get_chunknames(filename):
    dirname = os.path.dirname(filename)
    candidates = [os.path.join(dirname, x) for x in
                  natsort.natsorted(os.listdir(dirname))]
    return [x for x in candidates if is_chunkname(x, filename)]


def splitchunks_by_size(filename, chunksize, mem_usage=int(1e9)):
    chunksize, mem_usage = int(chunksize), int(mem_usage)
    file_remaining = os.stat(filename).st_size
    numchunks = ceil(file_remaining / chunksize)
    with open(filename, "rb") as f:
        for i_chunk in range(numchunks):
            with open(get_chunkname(filename, i_chunk), "wb") as chunk:
                chunk_remaining = chunksize
                nummems = min(file_remaining, chunksize) / mem_usage
                nummems = ceil(nummems)
                for _ in range(nummems):
                    writesize = min(mem_usage, chunk_remaining, file_remaining)
                    chunk.write(f.read(writesize))
                    file_remaining -= writesize
                    chunk_remaining -= writesize


def splitchunks_by_num(filename, numchunks, mem_usage=int(1e9)):
    size = os.stat(filename).st_size
    chunksize = ceil(size / int(numchunks))
    splitchunks_by_size(filename, chunksize, mem_usage=mem_usage)


def joinchunks(filename, mem_usage=int(1e9), rmchunks=False):
    mem_usage = int(mem_usage)
    chunknames = get_chunknames(filename)
    with open(filename, "wb") as f:
        for chunkname in chunknames:
            size = os.stat(chunkname).st_size
            with open(chunkname, "rb") as chunk:
                nummems = ceil(size / mem_usage)
                for _ in range(nummems):
                    f.write(chunk.read(min(mem_usage, size)))
    if rmchunks:
        for chunkname in chunknames:
            os.remove(chunkname)
