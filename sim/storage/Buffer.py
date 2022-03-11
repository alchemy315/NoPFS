from storage.CacheEntry import CacheEntry


class Buffer:

    def __init__(self, level, capacity):
        self.level = level
        self.capacity = capacity
        self.entries = {}
        self.curr_size = 0

    def can_add(self, file_size):
        return self.curr_size + file_size <= self.capacity

    def add(self, entry: CacheEntry):
        if self.curr_size + entry.size > self.capacity:
            raise ValueError("Buffer size exceeded")

        if entry.id in self.entries:
            curr_entries = self.entries[entry.id]
            if entry not in curr_entries:
                curr_entries.append(entry)
        else:
            self.entries[entry.id] = [entry]
            self.curr_size += entry.size

    def remove(self, entry: CacheEntry):
        if entry.id not in self.entries:
            return
        if len(self.entries[entry.id]) == 0:
            del self.entries[entry.id]
        else:
            entries = self.entries[entry.id]
            entries.remove(entry)
        self.curr_size -= entry.size

    def remove_by_id(self, id):
        if id not in self.entries:
            return
        size = len(self.entries[id]) * self.entries[id][0].size
        self.curr_size -= size

    def remove_by_id_and_batch(self, id, batch_no, batch_offset):
        if id not in self.entries:
            return
        entries = self.entries[id]
        for entry in entries:
            if entry.batch_no == batch_no and entry.batch_offset == batch_offset:
                entries.remove(entry)
                self.curr_size -= entry.size
                break

    def get_by_id(self, id):
        if id in self.entries:
            return self.entries[id]
        return []

    def get_by_id_and_batch(self, id, batch_no, batch_offset):
        entries = self.get_by_id(id)
        for entry in entries:
            if entry.batch_no is None or entry.batch_no == batch_no and entry.batch_offset == batch_offset:
                return entry
        return None