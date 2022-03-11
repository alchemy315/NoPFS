class CacheEntry:
    def __init__(self, id, avail_from, size, batch_no, batch_offset, source):
        self.id = id
        self.avail_from = avail_from
        self.read_counter = 0
        self.request_counter = 0
        self.size = size
        self.batch_no = batch_no
        self.batch_offset = batch_offset
        self.source = source

    def read(self):
        self.read_counter += 1

    def request(self):
        self.request_counter += 1

    def consumed(self):
        return self.read_counter > 0