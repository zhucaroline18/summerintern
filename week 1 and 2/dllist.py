class DoubleLinkedListNode(object):
    def __init__(self, value, nxt, before):
        self.value = value
        self.next = nxt
        self.before = before


class DoubleLinkedList(object):
    def __init__ (self):
        self.head = None

    def push(self, obj):
        newNode = DoubleLinkedListNode(obj, None, None)
        if self.head is None:
            self.head = newNode
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = newNode
            newNode.before = current

    def dump(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next
    
    def count(self):
        counts = 0
        current = self.head
        while current is not None:
            counts+=1
            current = current.next
        return counts
    
    def get(self, index):
        current = self.head
        for i in range(0,index):
            current = current.next
        return current.value
    
    def remove(self, obj):
        current = self.head
        currentVal = current.value
        
        while (currentVal!=obj):
            lastNode = current
            current = current.next
            currentVal = current.value
        nextNode = current.next
        lastNode.next = nextNode
        nextNode.before = lastNode