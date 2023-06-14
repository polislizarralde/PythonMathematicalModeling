class Modelo:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def solve(self):
        return self.nombre + " " + str(self.edad)


m = Modelo("Juan", 20)
print(m.solve())
print(m.save())
