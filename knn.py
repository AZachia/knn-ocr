import random
import json

class KNN:
    def __init__(self, k: int=3, elements: list=[], label_name:str = 'label') -> None:
        self.k = k
        self.elements = elements
        self.label_name = label_name
        
    def distance(self, e, element) -> float:
        communs = []
        for data in e:
            if data in element and data != self.label_name:
                communs.append(data)
        
        return (sum([(e[data]-element[data])**2 for data in communs]))**0.5
    
    
    def find_nearests(self, element: dict, keep_distance: bool = False) -> list:
        distances = []
        for e in self.elements:
            distances.append((e, self.distance(e, element)))
        distances.sort(key=lambda x: x[1])
        if keep_distance:
            return distances[:self.k]
        return [e[0] for e in distances[:self.k]]
        
    def predict(self, element: dict) -> str:
        labels = {}
        for el in self.find_nearests(element, False):
            if el[self.label_name] in labels:
                labels[el[self.label_name]] += 1
            else:
                labels[el[self.label_name]] = 1
        
        max_label, max_value = None, 0
        for label in labels:
            if labels[label] > max_value:
                max_label, max_value = label, labels[label]
        
        return max_label
    
    def predict_percent(self, element: dict) -> dict:
        labels = {}
        for el in self.find_nearests(element, False):
            if el[self.label_name] in labels:
                labels[el[self.label_name]] += 1
            else:
                labels[el[self.label_name]] = 1
        
        for label in labels:
            labels[label] /= self.k
        
        return labels
    
    def score(self, elements: list) -> float:
        correct = 0
        for el in elements:
            if self.determine(el) == el[self.label_name]:
                correct += 1
        return correct/len(elements)
    
    def regression(self, element: dict, precision: int = 3) -> float:
        proptietes = {}
        neigbours = self.find_nearests(element, True)
        
        for el, distance in neigbours:
            for propriete in el:
                if propriete != self.label_name and not propriete in element:
                    if distance != 0:
                        if propriete in proptietes:
                            proptietes[propriete].append((el[propriete], 1/distance))
                        else:
                            proptietes[propriete] = [(el[propriete], 1/distance)]
                    else:
                        if propriete in proptietes:
                            proptietes[propriete].append((el[propriete], 1))
                        else:
                            proptietes[propriete] = [(el[propriete], 1)]

        for propriete in proptietes:
            total_weight = sum(weight for _, weight in proptietes[propriete])
            if total_weight != 0:
                weighted_sum = sum(value * weight for value, weight in proptietes[propriete])
                proptietes[propriete] = round(weighted_sum / total_weight, precision)
            
        return {**element, **proptietes}

    
    def add_element(self, element: dict) -> None:
        self.elements.append(element)
    
    def remove_element(self, element: dict) -> None:
        self.elements.remove(element)
    
    def load_json(self, path: str = "knnset.json") -> None:
        with open(path, 'r') as file:
            self.elements = json.load(file)
    
    def save_json(self, path: str = "knnset.json") -> None:
        with open(path, 'w') as file:
            json.dump(self.elements, file)
    
    def load_csv(self, path: str = "knnset.csv", sep: str = ";") -> None:
        with open(path, 'r') as file:
            lines = file.readlines()
            labels = lines[0].split(sep)
            for line in lines[1:]:
                data = line.split(sep)
                self.elements.append({labels[i]: data[i] for i in range(len(labels))})
    
    def save_csv(self, path: str = "knnset.csv", sep: str = ";") -> None:
        
        raise NotImplementedError("Not implemented yet")
        
        with open(path, 'w') as file:
            ...
    
    def __call__(self, element: dict) -> str:
        return self.predict(element)
        # usage: knn(element)

    def __getitem__(self, element: dict):
        return self.predict(element)
        # usage: knn[element]

    def __setitem__(self, element: dict):
        self.add_element(element)
        # usage: knn[element] = element

        

def generate_elements(template: dict, size: int | tuple= 50, precision: int = 10) -> list:
    if type(size) == tuple:
        nb_elements = random.randint(*size)
    else:
        nb_elements = size
    
    elements = []
    for el in range(nb_elements):
        new_el = {}
        for categorie in template:
            if type(template[categorie]) == tuple:
                a, b = template[categorie][0]*precision, template[categorie][1]*precision
                new_el[categorie] = random.randint(a, b)/precision
            else:
                new_el[categorie] = template[categorie]
        elements.append(new_el)
    return elements
                
        
    

if __name__ == "__main__":
    elements = [
        {'label': 'stylo', "taille": 10, "diametre": 0.4},
        {'label': 'stylo', "taille": 15.4, "diametre": 0.5},
        {'label': 'stylo', "taille": 13.5, "diametre": 0.3},
        {'label': 'surlinieur', "taille": 6, "diametre": 2},
        {'label': 'surlinieur', "taille": 7.4, "diametre": 1.5},
        {'label': 'surlinieur', "taille": 5.2, "diametre": 2.3},
        {'label': 'gomme', "taille": 3, "diametre": 4},
        {'label': 'gomme', "taille": 2, "diametre": 5},
        {'label': 'gomme', "taille": 4, "diametre": 3}
    ]
    
    knn = KNN(elements=elements)
    print(knn.regression({"taille": 6.4}))
    
    knn.save_json()
    
    # print(generate_elements({'label': 'gomme', "taille": (2, 5), "diametre": (1, 4)}))
    