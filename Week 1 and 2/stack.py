class StackListNode(object):
    def __init__(self, value, next):
        self.value = value
        self.next = next


class StackList(object):
    def __init__ (self):
        self.head = None
    
    def push(self, obj):
        newNode = StackListNode(obj, None)
        if self.head is None:
            self.head = newNode
        else:
            newNode.next = self.head
            self.head = newNode

    def pop(self):
        value = self.head.value
        self.head = self.head.next
        return value
        
    def dump(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next