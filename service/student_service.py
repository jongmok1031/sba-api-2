
class StudentService:
    def __init__(self):
        self.dao = StudentDao()

    def add_student(self, student):
        print('### add_student ###')
        self.dao.create()
        self.dao.insert_many()
        print(f'>>> 입력된 학생들의 수: {self.dao.fetch_count()}')

    def login(self, id, pwd):
        return self.dao.login(id, pwd)
