import sllist

def test_push():
    list = sllist.SingleLinkedList()
    list.push(1)
    list.push(2)
    list.dump()

def test_get():
    list = sllist.SingleLinkedList()
    list.push(0)
    list.push(1)
    list.push(2)
    list.push(3)
    list.push(4)
    x=list.get(4)
    print (x)

if __name__ == "__main__":
    test_get()
    
