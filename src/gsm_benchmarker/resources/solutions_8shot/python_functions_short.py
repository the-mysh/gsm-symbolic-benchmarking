
SOLUTIONS = []


def solution():
    trees_now = 15
    trees_after_planting = 21

    # calculate the difference between trees after planting and trees originally
    trees_to_plant_today = trees_after_planting - trees_now
    return trees_to_plant_today


SOLUTIONS.append(solution)


def solution():
    cars_before = 3
    cars_arriving = 2

    # calculate the sum of cars originally in the parking lot and the arriving cars
    cars_after = cars_before + cars_arriving
    return cars_after


SOLUTIONS.append(solution)


def solution():
    chocolates_leah_before = 32
    chocolates_sister_before = 42
    chocolates_eaten = 35

    # first, calculate how many chocolates the sisters originally had in total
    chocolates_total_before = chocolates_leah_before + chocolates_sister_before
    # next, subtract the total number of chocolates they ate
    chocolates_left = chocolates_total_before - chocolates_eaten
    return chocolates_left


SOLUTIONS.append(solution)


def solution():
    lollipops_jason_before = 20
    lollipops_jason_now = 12

    # calculate the difference between lollipops Jason had originally and lollipops he has now
    lollipops_given_to_denny = lollipops_jason_before - lollipops_jason_now
    return lollipops_given_to_denny


SOLUTIONS.append(solution)


def solution():
    shawn_toys_before = 5
    new_toys_from_each_parent = 2

    # first, calculate how many toys in total Shawn got for Christmas
    new_toys_total = 2 * new_toys_from_each_parent
    # next, calculate the sum of the number of toys Shawn originally had and the number of new toys he got
    shawn_toys_now = shawn_toys_before + new_toys_total
    return shawn_toys_now


SOLUTIONS.append(solution)


def solution():
    computers_before = 9
    computers_installed_per_day = 5
    number_of_days = 4

    # first, calculate how many new computers were installed in total
    computers_installed_total = number_of_days * computers_installed_per_day
    # next, calculate sum of the number of computers originally in the room and the total number of new computers installed
    computers_now = computers_before + computers_installed_total
    return computers_now


SOLUTIONS.append(solution)


def solution():
    golf_balls_before = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2

    # first, calculate the total number of golf balls Michael lost
    golf_balls_lost_total = golf_balls_lost_tuesday + golf_balls_lost_wednesday
    # next, calculate the difference between the original number of golf balls and the number of golf balls lost
    golf_balls_left = golf_balls_before - golf_balls_lost_total
    return golf_balls_left


SOLUTIONS.append(solution)


def solution():
    dollars_before = 23
    bagels = 5
    price_per_bagel = 3

    # first, calculate the total cost of the bagels
    bagels_cost_total = bagels * price_per_bagel
    # next, calculate the difference between Olivia's original money and what she paid for the bagels
    dollars_left = dollars_before - bagels_cost_total
    return dollars_left


SOLUTIONS.append(solution)
