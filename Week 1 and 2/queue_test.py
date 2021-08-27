import myqueue

def test_queue():
    thing = myqueue.QueueList()
    thing.push(2)
    thing.push(3)
    thing.push(4)
    thing.dump()
    assert thing.pop()==2
    thing.dump()

if __name__ == "__main__":
    test_queue()

