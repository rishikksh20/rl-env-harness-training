def chunk_items(items, size):
    stop = (len(items) // size) * size
    return [items[index : index + size] for index in range(0, stop, size)]
