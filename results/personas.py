from enum import Enum

class Persona(Enum):
    Adele = "adele"
    Bale = "bale"
    Beckham = "beckham"
    Campbell = "campbell"
    Craig = "craig"
    Cumberbatch = "cumberbatch"
    Delevingne = "delevingne"
    Dench = "dench"
    Gervais = "gervais"
    Grant = "grant"
    Hamilton = "hamilton"
    Jagger = "jagger"
    John = "john"
    Middleton = "middleton"
    Moss = "moss"
    Radcliffe = "radcliffe"
    Rowling = "rowling"
    Sheeran = "sheeran"
    Smith = "smith"
    Watson = "watson"

    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, Persona):
            return self.value == other.value
        return False
    
    def isHimIsHer(self, name: str):
        return name.lower() == self.value
