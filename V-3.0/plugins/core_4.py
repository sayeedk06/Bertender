#euclideandistance
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')

class Plugin:
    def __init__(self,*args):
        print('\t\n****Seondary window plugin activated****\n')

    #Finding the remaining sentences starts here

    def source_of_words(self,the_remaining_coordinates, sentence_dictionary):
        sentences = []
        for key, value in sentence_dictionary.items():
            for i in value:
                for j in the_remaining_coordinates:
                    if i == j:
                        sentences.append(key)
        return sentences

    #Finding the remaining sentences ends here

    #Finding word instances starts here
    def word_plot(self,label,x,y,word):
        # print("Initial:\n")
        # print(x)
        # print(y)
        indexforelist = []
        countfore = 0
        for i in label:
            if i == word:
                indexforelist.append(countfore)
            countfore +=1
        # print(indexforelist)
        mappingxco_ordinates = []
        mappingyco_ordinates = []

        for i in indexforelist:
            mappingxco_ordinates.append(x[i])
            mappingyco_ordinates.append(y[i])

        # print("function")
        # print(mappingxco_ordinates)
        # print(mappingyco_ordinates)
        return mappingxco_ordinates , mappingyco_ordinates
    #Finding word instances ends here

    #Dictionary of the euclidean distances starts here
    def distance_x_y(self,label,x,y,word):
        indexforelist = []
        countfore = 0

        for i in label:
            if i == word:
                indexforelist.append(countfore)
            countfore +=1

        mappingxco_ordinates = []
        mappingyco_ordinates = []

        for i in indexforelist:
            mappingxco_ordinates.append(x[i])
            mappingyco_ordinates.append(y[i])

        length_required = len(mappingxco_ordinates)

        xypairedlist = []

        for i in range (0,length_required):
            k = []
            k.append(mappingxco_ordinates[i])
            k.append(mappingyco_ordinates[i])
            xypairedlist.append(k)

        a = dist.pairwise(xypairedlist)

        count = 0
        distance_list = []

        for i in a:
            for j in i:
                count += 1
                distance_list.append(j)

        listco = [] #a list in the order the distances are shown

        for i in xypairedlist:
            for j in xypairedlist:
                listco.append(i)
                listco.append(j)

        distance_dictionary = dict()

        for index, item in enumerate(distance_list):
            target_start_index = 2 * index
            target_end_index = 2 * index + 1

            distance_dictionary[item] = listco[target_start_index:(target_end_index + 1)]



        return distance_dictionary, distance_list
    #Dictionary of the euclidean distances ends here

    #Plotting the words outside a certain boundary starts here
    def boundary_word_plot(self, dictionary, distance_list, boundary):

        the_two_boundaries = []

        for i in dictionary:
            if i > boundary:

                the_two_boundaries.append(dictionary[i])

        therestx = []
        theresty = []

        for i in the_two_boundaries:
            for j in i:
                therestx.append(j[0])
                theresty.append(j[1])

        thecoordinates = []
        for i in the_two_boundaries:
            for j in i:
                thecoordinates.append(j)

        return therestx, theresty, thecoordinates
    #Plotting the words outside a certain boundary ends here
