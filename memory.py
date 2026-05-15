memory_store = {}

def save_memory(user, message):
    if user not in memory_store:
        memory_store[user] = []

    memory_store[user].append(message)

def get_memory(user):
    return memory_store.get(user, [])