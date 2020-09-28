from dataclasses import dataclass
'''
context path : /Users/jongm/SBAprojects
fname : /kaggle/data
'''
@dataclass
class FileReader:
    context: str = ''
    fname: str = ''
    train : object = None
    test : object = None
    id : str = ''
    label : str = ''
    # 전형적인 지도학습 entity
    