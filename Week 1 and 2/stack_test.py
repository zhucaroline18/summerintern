import stack

def test_stackpop():
    thing = stack.StackList()
    thing.push(2)
    thing.push(3)
    thing.push(4)
    thing.dump()
    assert thing.pop()==4
    thing.dump()

if __name__ == "__main__":
    test_stackpop()

