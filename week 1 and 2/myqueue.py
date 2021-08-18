class QueueListNode:
    def __init__(self, value, nxt):
        self.value = value
        self.next = nxt

class QueueList:
    def __init__(self):
        self.head = None

    def push (self, obj):
        newNode = QueueListNode(obj, None)
        if self.head is None:
            self.head = newNode
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = newNode

    def pop(self):
        value = self.head.value
        self.head = self.head.next
        return value

    def dump(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next