box0 = []
box1 = []
box2 = []
box3 = []
box4 = []
box5 = []
box6 = []
box7 = []
box8 = []
box9 = []

def getHash(x):
    sum = 0
    for c in x:
        sum+= ord(c)
    return sum%10
    
def assignBox(x):
    hashNumber = getHash(x)
    remain = hashNumber%10
    if remain == 0:
        box0.append(x)
    elif remain == 1:
        box1.append(x)
    elif remain == 2:
        box2.append(x)
    elif remain == 3:
        box3.append(x)
    elif remain == 4:
        box4.append(x)
    elif remain == 5:
        box5.append(x)
    elif remain == 6:
        box6.append(x)
    elif remain == 7:
        box7.append(x)
    elif remain == 8:
        box8.append(x)
    elif remain == 9:
        box9.append(x)
        

class MyDictNode:
    def __init__ (self, key, value, next):
        self.key = key
        self.value = value
        self.next = next


class MyDictNodeList:
    def __init__(self):
        self.head = None

    def set(self, key, val):
        if self.head is None:
            self.head = MyDictNode(key, val, None)
        else:
            existingNode = None

            current = self.head
            while current is not None:
                if current.key == key:
                    existingNode = current
                    break
                current = current.next

            if existingNode is None:
                self.head = MyDictNode(key, val, self.head)
            else:
                existingNode.value = val

    def get(self, key):
        existingNode = None
        current = self.head 
        while current is not None:
            if current.key == key:
                existingNode = current
                break
            current = current.next 
        
        if existingNode is None:
            return None
        else:
            return existingNode.value

class MyDict:
    def __init__(self): 
        self.bucketCount = 100

        self.buckets = [MyDictNodeList() for x in range(self.bucketCount)]
        
    def getBucketIndex(self,key):
        return hash(key)%self.bucketCount

    def set(self, key, value):
        bucketIndex = self.getBucketIndex(key)
        self.buckets[bucketIndex].set(key, value)

    def get(self, key):
        bucketIndex = self.getBucketIndex(key)
        return self.buckets[bucketIndex].get(key)