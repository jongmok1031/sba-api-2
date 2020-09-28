import sqlite3

class Student:
    id: str = ''
    pwd: str = ''
    name: str = ''
    birth: str = ''
    regdate: str = ''


# DAO : Database Access Object
class StudentDao:

    def __init__(self):
        self.conn = sqlite3.connect('sqlite.db')
        self.cursor = self.conn.cursor()

    def create(self):
        # cursor(커서) : 데이터베이스 작업을 수행하고 있는 메모리 공간
        cursor = self.cursor
        try:
            # excute : sql 구문을 실행해주는 함수
            cursor.execute("drop table students")
        except sqlite3.OperationalError as err:
            print("테이블이 존재하지 않습니다.")

        cursor.execute(
            '''create table students
            (id text primary key, pwd text, name text, birth text)'''
        )
        self.conn.commit()

    def insert_one(self, student):
        cursor = self.cursor
        sql = """
            insert into students(id, pwd, name, birth) values (?, ?, ?, ?)
        """
        cursor.execute(sql, (student.id, student.pwd, student.name, student.birth))
        self.conn.commit()

    def insert_many(self):
        cursor = self.cursor
        data = [('jo','1', '조용필', '1985/12/31'), ('ko','1', '고아라', '1970/07/17'), ('sim','1', '심형래', '1950/06/06')]
        # ?: place holder : 치환되어야 할 어떤 대상
        # 데이터 유형에 상관없이 외따옴표 적지 마라
        sql = """
            insert into students(id, pwd, name, birth) values (?, ?, ?, ?)
        """
        cursor.executemany(sql, data)
        self.conn.commit()

    def fetch_by_id(self, id):
        cursor = self.cursor
        sql = "select * from students where id = '%s'"
        cursor.execute(sql, (id))
        result = cursor.fetchone() # fetch 해줘야 함
        print(type(result)) # 튜플 형태로 리턴
        if result != None:
            print('아이디 : ' + result[0], end='')
            print(', 이름 : ' + result[1], end='')
            print(', 생일 : ' + result[2], end='')
        else:
            print('문제가 있습니다.')
        return result

    def fetch_list(self):
        cursor = self.cursor
        sql = """select * from students order by name desc"""
        for id, name, birth in cursor.execute(sql): # 다중 데이터는 for문으로 바로 출력가능
            print(id + '#' + name + '#' + birth)
        print('-'*20)
        cursor.execute(sql)
        return cursor.fetchall()

    def fetch_by_name(self, name):
        cursor = self.cursor
        sql = """
            select * from students where name like '%?%'
        """
        cursor.execute(sql, (name))
        return cursor.fetchall() 

    def fetch_count(self):
        cursor = self.cursor
        sql = """select count(*) from students"""
        cursor.execute(sql)
        return cursor.fetchone()

    def fetch_all(self):
        cursor = self.cursor
        cursor.execute('select * from students')
        return cursor.fetchall()

    def login(self, id, pwd):
        cursor = self.cursor
        sql = """
        select * from  students where id like ? and pwd like ?
        """
        cursor.execute(sql, (id, pwd))
        temp = cursor.fetchone()
        return temp[2]
        
    def update(self, id, name):
        # 'id'가 'lee'인 친구의 이름을 '이문세'로 바꾸세요.
        cursor = self.cursor
        sql = """update students set name = ? where id = ?"""
        cursor.execute(sql, (id, name))
        print(cursor.rowcount) # 성공 여부
        self.conn.commit()

    def delete(self, id):
        # 'id' 데이터를 삭제하세요.
        cursor = self.cursor
        sql = """delete from students where id = ?"""
        cursor.execute(sql, (id))
        print(cursor.rowcount)
        self.conn.commit()

        # cursor.close()
        # conn.close() web 상에서는 close 하지 않음