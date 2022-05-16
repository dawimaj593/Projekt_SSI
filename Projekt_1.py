import pandas as pd
import random
import numpy as np


class Data_Processing:
    @staticmethod
    def ShuffleData(set_dataframe):
        for i in range(0, len(set_dataframe) - 1, 1):
            j = random.randint(0, i)
            set_dataframe.iloc[i], set_dataframe.iloc[j] = set_dataframe.iloc[j], set_dataframe.iloc[i]
        return set_dataframe

    @staticmethod
    def SplitData(set_dataframe):
        division = len(set_dataframe) / 10 * 7
        train_set = set_dataframe[:int(division)]
        val_set = set_dataframe[int(division + 1):]
        return train_set, val_set

# klasa do rozpoznawania formcaji
class Price_action:
    @staticmethod
    def hammer(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_max_price = set_dataframe.loc[index, "Najwyzszy"]
            formation_min_price = set_dataframe.loc[index, "Najnizszy"]
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_open_price < formation_close_price and 4 * (formation_close_price - formation_open_price) <=(formation_max_price - formation_min_price):
                if formation_open_price > (formation_max_price + formation_min_price) / 2:
                    if index > 3 and index < len(set_dataframe) - 3:
                        list = []
                        before_open_price = set_dataframe.loc[index-2, "Otwarcie"]
                        before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                        after_close_price = set_dataframe.loc[index+1, "Zamkniecie"]

                        list.append(abs(before_open_price - formation_open_price))
                        if before_open_price <= before_close_price:
                            list.append("spadek")
                        else:
                            list.append("wzrost")
                        list.append("hammer")
                        if formation_close_price < after_close_price:
                            list.append("wzrost")
                        else:
                            list.append("spadek")
                        list_of_cases.append(list)
        return list_of_cases

    @staticmethod
    def marubozu(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_max_price = set_dataframe.loc[index, "Najwyzszy"]
            formation_min_price = set_dataframe.loc[index, "Najnizszy"]
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_close_price - formation_open_price > 30:
                if formation_max_price - formation_close_price < 3 and formation_min_price - formation_open_price < 3:
                    if index > 3 and index < len(set_dataframe) - 3:
                        list = []
                        before_open_price = set_dataframe.loc[index - 2, "Otwarcie"]
                        before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                        after_close_price = set_dataframe.loc[index + 1, "Zamkniecie"]

                        list.append(abs(before_open_price - formation_open_price))
                        if before_open_price <= before_close_price:
                            list.append("spadek")
                        else:
                            list.append("wzrost")
                        list.append("marubozu")
                        if formation_close_price < after_close_price:
                            list.append("wzrost")
                        else:
                            list.append("spadek")
                        list_of_cases.append(list)
        return list_of_cases

    @staticmethod
    def engulfing_Pattern_hossa(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_close_price < formation_open_price:
                if index > 3 and index < len(set_dataframe) - 3:
                    formation_open_price_2 = set_dataframe.loc[index + 1, "Otwarcie"]
                    formation_close_price_2 = set_dataframe.loc[index + 1, "Zamkniecie"]

                    if formation_open_price < formation_close_price_2 and formation_open_price_2 < formation_close_price:
                            list = []
                            before_open_price = set_dataframe.loc[index - 2, "Otwarcie"]
                            before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                            after_close_price = set_dataframe.loc[index + 1, "Zamkniecie"]

                            list.append(abs(before_open_price - formation_open_price))
                            if before_open_price <= before_close_price:
                                list.append("spadek")
                            else:
                                list.append("wzrost")
                            list.append("engulfing_Pattern_hossa")
                            if formation_close_price_2 < after_close_price:
                                list.append("wzrost")
                            else:
                                list.append("spadek")
                            list_of_cases.append(list)
        return list_of_cases

    @staticmethod
    def engulfing_Pattern_bessa(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_close_price > formation_open_price:
                if index > 3 and index < len(set_dataframe) - 3:
                    formation_open_price_2 = set_dataframe.loc[index + 1, "Otwarcie"]
                    formation_close_price_2 = set_dataframe.loc[index + 1, "Zamkniecie"]
                    if formation_open_price > formation_close_price_2 and formation_open_price_2 < formation_close_price:
                            list = []
                            before_open_price = set_dataframe.loc[index - 2, "Otwarcie"]
                            before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                            after_close_price = set_dataframe.loc[index + 1, "Zamkniecie"]

                            list.append(abs(before_open_price - formation_open_price))
                            if before_open_price <= before_close_price:
                                list.append("spadek")
                            else:
                                list.append("wzrost")
                            list.append("engulfing_Pattern_bessa")
                            if formation_close_price_2 < after_close_price:
                                list.append("wzrost")
                            else:
                                list.append("spadek")
                            list_of_cases.append(list)
        return list_of_cases

    @staticmethod
    def three_white_soldiers(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_close_price > formation_open_price:
                if index > 3 and index < len(set_dataframe) - 4:
                    formation_open_price_2 = set_dataframe.loc[index + 1, "Otwarcie"]
                    formation_close_price_2 = set_dataframe.loc[index + 1, "Zamkniecie"]

                    if formation_close_price > formation_open_price_2 and formation_close_price_2 > formation_close_price:
                        formation_open_price_3 = set_dataframe.loc[index + 2, "Otwarcie"]
                        formation_close_price_3 = set_dataframe.loc[index + 2, "Zamkniecie"]

                        if formation_close_price_2 > formation_open_price_3 and formation_close_price_3 > formation_close_price_2:
                            list = []
                            before_open_price = set_dataframe.loc[index - 2, "Otwarcie"]
                            before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                            after_close_price = set_dataframe.loc[index + 3, "Zamkniecie"]

                            list.append(abs(before_open_price - formation_open_price))
                            if before_open_price <= before_close_price:
                                list.append("spadek")
                            else:
                                list.append("wzrost")
                            list.append("three_white_soldiers")
                            if formation_close_price_3 < after_close_price:
                                list.append("wzrost")
                            else:
                                list.append("spadek")
                            list_of_cases.append(list)
        return list_of_cases

    @staticmethod
    def three_black_ravens(set_dataframe):
        list_of_cases = []
        for index, row in set_dataframe.iterrows():
            formation_open_price = set_dataframe.loc[index, "Otwarcie"]
            formation_close_price = set_dataframe.loc[index, "Zamkniecie"]

            if formation_close_price < formation_open_price:
                if index > 3 and index < len(set_dataframe) - 4:
                    formation_open_price_2 = set_dataframe.loc[index + 1, "Otwarcie"]
                    formation_close_price_2 = set_dataframe.loc[index + 1, "Zamkniecie"]

                    if formation_close_price < formation_open_price_2 and formation_close_price_2 < formation_close_price:
                        formation_open_price_3 = set_dataframe.loc[index + 2, "Otwarcie"]
                        formation_close_price_3 = set_dataframe.loc[index + 2, "Zamkniecie"]

                        if formation_close_price_2 < formation_open_price_3 and formation_close_price_3 < formation_close_price_2:
                            list = []
                            before_open_price = set_dataframe.loc[index - 2, "Otwarcie"]
                            before_close_price = set_dataframe.loc[index - 2, "Zamkniecie"]
                            after_close_price = set_dataframe.loc[index + 3, "Zamkniecie"]

                            list.append(abs(before_open_price - formation_open_price))
                            if before_open_price <= before_close_price:
                                list.append("spadek")
                            else:
                                list.append("wzrost")
                            list.append("three_black_ravens")
                            if formation_close_price_3 < after_close_price:
                                list.append("wzrost")
                            else:
                                list.append("spadek")
                            list_of_cases.append(list)
        return list_of_cases

# Klasyfikator Bayesa
class Naive_bayes:
    @staticmethod
    def average(series_training_set):
        number_of_samples = len(series_training_set)
        sum = 0

        for i in series_training_set:
            sum += i

        avg = sum/number_of_samples

        return avg

    @staticmethod
    def std(series_training_set, avg):
        number_of_samples = len(series_training_set)
        sum = 0

        for i in series_training_set:
            sum += (avg - i)**2

        return (sum/number_of_samples)**(0.5)

    @staticmethod
    def gauss(series_training_set, avg, std):
        gauss_list = []
        for i in series_training_set:
            x = np.exp(-(i-avg)**2/(2*std**2))
            z = 1/np.sqrt(2*np.pi*std**2)*x
            gauss_list.append(z)

        return gauss_list

    @staticmethod
    def classify(data_traning_set, data_validations_set):
        # separacja klas
        increase_database = data_traning_set[data_traning_set.kierunek_po_formacji == "wzrost"]
        decrease_database = data_traning_set[data_traning_set.kierunek_po_formacji == "spadek"]

        increase_database_val = data_validations_set[data_validations_set.kierunek_po_formacji == "wzrost"]
        decrease_database_val = data_validations_set[data_validations_set.kierunek_po_formacji == "spadek"]

        # liczenie składowych prawdopodobienstw dla kontynuacji wzrostowych
        increase_direction_before = increase_database["kierunek_przed_formacja"]
        increase_formation = increase_database["formacja"]

        # liczenie składowych prawdopodobienstw dla kontynuacji spadkowych
        decrease_direction_before = decrease_database["kierunek_przed_formacja"]
        decrease_formation = decrease_database["formacja"]

        # Gauss dla bazy walidacyjnej
        val_mean = Naive_bayes.average(data_traning_set["ruch"])
        val_std = Naive_bayes.std(data_traning_set["ruch"], val_mean)
        validation_gauss_list = Naive_bayes.gauss(data_validations_set["ruch"], val_mean, val_std)

        corect_answer = 0

        # klasyfikacja
        for index, row in data_validations_set.iterrows():
            if row["kierunek_po_formacji"] == "wzrost":
                direction_before_val = row["kierunek_przed_formacja"]
                formation_val = row["formacja"]
                direction_after_val = row["kierunek_po_formacji"]
                gauss_val = validation_gauss_list.pop(0)

                direction_before_val_counter = 0
                formation_val_counter = 0

                # sprawdzanie klasy 'wzrost'
                for x in increase_direction_before:
                    if x == direction_before_val:
                        direction_before_val_counter += 1

                for x in increase_formation:
                    if x == formation_val:
                        formation_val_counter += 1

                # liczenie prawdopodobienstwa
                increase_probability = gauss_val * (direction_before_val_counter / len(increase_database_val) *
                                                    formation_val_counter/len(increase_database))/2

                direction_before_val_counter = 0
                formation_val_counter = 0

                # sprawdzanie dla spadków
                for x in decrease_direction_before:
                    if x == direction_before_val:
                        direction_before_val_counter += 1

                for x in decrease_formation:
                    if x == formation_val:
                        formation_val_counter += 1

                decrease_probability = gauss_val * (direction_before_val_counter / len(decrease_database_val) *
                                                    formation_val_counter / len(decrease_database))/2

                if increase_probability >= decrease_probability:
                    answer = "wzrost"
                else:
                    answer = "spadek"

                if answer == direction_after_val:
                    corect_answer += 1
            else:
                direction_before_val = row["kierunek_przed_formacja"]
                formation_val = row["formacja"]
                direction_after_val = row["kierunek_po_formacji"]
                gauss_val = validation_gauss_list.pop(0)

                direction_before_val_counter = 0
                formation_val_counter = 0

                # sprawdzanie klasy 'wzrost'
                for x in increase_direction_before:
                    if x == direction_before_val:
                        direction_before_val_counter += 1

                for x in increase_formation:
                    if x == formation_val:
                        formation_val_counter += 1

                # liczenie prawdopodobienstwa
                increase_probability = gauss_val * (direction_before_val_counter / len(increase_database_val)
                                                    * formation_val_counter / len(increase_database)) / 2

                direction_before_val_counter = 0
                formation_val_counter = 0

                # sprawdzanie dla spadków
                for x in decrease_direction_before:
                    if x == direction_before_val:
                        direction_before_val_counter += 1

                for x in decrease_formation:
                    if x == formation_val:
                        formation_val_counter += 1

                decrease_probability = gauss_val * (direction_before_val_counter / len(decrease_database_val) *
                                                    formation_val_counter / len(decrease_database)) / 2

                if increase_probability >= decrease_probability:
                    answer = "wzrost"
                else:
                    answer = "spadek"

                if answer == direction_after_val:
                    corect_answer += 1

        print("Skutecznosc: " + str(corect_answer/len(data_validations_set)) + "%")
        print("Poprawne: " + str(corect_answer) + "/" + str(len(data_validations_set)))


# --------------------------------------------------------------------------------------
# importowanie bazy notowań wig20
data_base_wig_20 = pd.read_csv("wig20_db.csv")

# tworzenie list wszystkich wystąpien formacji, ruchu cen po i przed formacją
formation_list = [Price_action.hammer(data_base_wig_20),
                  Price_action.marubozu(data_base_wig_20),
                  Price_action.engulfing_Pattern_hossa(data_base_wig_20),
                  Price_action.engulfing_Pattern_bessa(data_base_wig_20),
                  Price_action.three_white_soldiers(data_base_wig_20),
                  Price_action.three_black_ravens(data_base_wig_20)]

# konwertowanie list do jednego DataFrame do przeanalizowania przez algorytm Bayesa
# usuwanie pustych formacji z listy
to_pop = []
for j in range(len(formation_list)):
    if len(formation_list[j]) == 0:
        to_pop.append(j)
for k in to_pop:
    formation_list.pop(k)

# łączenie formacji:
if len(formation_list) > 1:
    formation_list1 = pd.DataFrame(formation_list.pop(0))
    formation_list1.columns = 'ruch', 'kierunek_przed_formacja', 'formacja', 'kierunek_po_formacji'
    formation_list2 = pd.DataFrame(formation_list.pop(1))
    formation_list2.columns = 'ruch', 'kierunek_przed_formacja', 'formacja', 'kierunek_po_formacji'
    data_base_formation = pd.merge(formation_list1, formation_list2, how='outer')
    for i in range(len(formation_list)):
        formation_list_exp = pd.DataFrame(formation_list[i])
        formation_list_exp.columns = 'ruch', 'kierunek_przed_formacja', 'formacja', 'kierunek_po_formacji'
        data_base_formation = pd.merge(data_base_formation, formation_list_exp, how='outer')
elif len(formation_list) == 1:
    data_base_formation = pd.DataFrame(formation_list)
    data_base_formation.columns = 'ruch', 'kierunek_przed_formacja', 'formacja', 'kierunek_po_formacji'
else:
    print("Wystąpił problem z bazą danych. Nie można dokonać klasyfikacji.")
    exit()


# mieszanie i podział stworzonego dataframe na zbiór walidacyjny i testowy
data_base_formation_shuffled = Data_Processing.ShuffleData(data_base_formation)
data_base_formation_train, data_base_formation_val = Data_Processing.SplitData(data_base_formation_shuffled)

# reszta kodu
Naive_bayes.classify(data_base_formation_train, data_base_formation_val)

