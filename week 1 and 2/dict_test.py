import dict

def test_mydict():
    d = dict.MyDict()
    d.set("a", "Caroline")
    d.set("a", "Caroline Zhu")
    d.set("b", "Cathleen")
    assert d.get("a")=="Caroline Zhu"
    assert d.get("b")=="Cathleen"
if __name__=="__main__":
    test_mydict()