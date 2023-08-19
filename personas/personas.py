from enum import Enum

"""
    Assumption: File names are unique and from the format: <persona_family_name>_<text_number>.txt
"""

class Persona(Enum):
    Adele = "adele"
    Bale = "bale"
    Beckham = "beckham"
    Campbell = "campbell"
    Craig = "craig"
    Cumberbatch = "cumberbatch"
    Delevigne = "delevigne"
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
    
    
    def __repr__(self):
        return self.value
    
    
    def __eq__(self, other):
        return isinstance(other, Persona) and self.value == other.value


    def get_optional_names(self):
        match self:
            case Persona.Adele:
                return ["Adele", "Adele Laurie Blue Adkins", "Adele Adkins"]
            case Persona.Bale:
                return ["Christian Bale"]
            case Persona.Beckham:
                return ["David Beckham"]
            case Persona.Campbell:
                return ["Naomi Campbell"]
            case Persona.Craig:
                return ["Daniel Craig"]
            case Persona.Cumberbatch:
                return ["Benedict Cumberbatch"]
            case Persona.Delevigne:
                return ["Cara Delevingne", "Delevingne"]
            case Persona.Dench:
                return ["Judi Dench"]
            case Persona.Gervais:
                return ["Ricky Gervais"]
            case Persona.Grant:
                return ["Hugh Grant"]
            case Persona.Hamilton:
                return ["Lewis Hamilton"]   
            case Persona.Jagger:
                return ["Mick Jagger"]
            case Persona.John:
                return ["Elton John"]
            case Persona.Middleton:
                return ["Kate Middleton"]
            case Persona.Moss:
                return ["Kate Moss"]
            case Persona.Radcliffe:
                return ["Daniel Radcliffe"]
            case Persona.Rowling:
                return ["J.K. Rowling",  "Joanne Rowling",  "Joanne Kathleen Rowling"]
            case Persona.Sheeran:
                return ["Ed Sheeran"]
            case Persona.Smith:
                return ["Sam Smith"]
            case Persona.Watson:
                return ["Emma Watson"]
            case _:
                raise Exception(f"Unknown persona {self.value}")