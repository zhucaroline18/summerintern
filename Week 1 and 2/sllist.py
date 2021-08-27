class SingleLinkedListNode(object):
    def __init__(self, value, next):
        self.value = value
        self.next = next


class SingleLinkedList(object):
    def __init__ (self):
        self.head = None

    def push(self, obj):
        newNode = SingleLinkedListNode(obj, None)
        if self.head is None:
            self.head = newNode
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = newNode


    def pop(self):
        """Removes the last item and returns it."""
    
    def shift(self, obj):
        """Another name for push."""
    
    def unshift(self):
        """Removes the first item and returns it."""
        
    def remove(self, obj):
        current = self.head
        currentVal = current.value
        
        while (currentVal!=obj):
            lastNode = current
            current = current.next
            currentVal = current.value
        nextNode = current.next
        lastNode.next = nextNode
        
    
    def first(self):
        """Returns a *reference* to the first item, does not remove."""

    def last(self):
        """Returns a reference to the last item, does not remove."""
    
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
            
    
    def dump(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next