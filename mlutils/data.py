

def chunk_to_batch(x, batch_size):
    return [
        x[i:i+batch_size]
        for i in range(0, len(x), batch_size)
    ]
