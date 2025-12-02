FUNCTIONS = []


def solution():
    # given:
    trees_now = 15  # number of trees currently in the grove
    trees_after_planting = 21  # number of trees in the grove after the workers are done planting today

    # to find: number of trees the grove workers will plant today

    # solution:
    # calculate the difference between trees after planting and trees originally
    trees_to_plant_today = trees_after_planting - trees_now
    return trees_to_plant_today


FUNCTIONS.append(solution)


def solution():
    # given:
    cars_originally = 3  # number of cars originally in the parking lot
    cars_arriving = 2  # number of cars arriving in teh parking loy

    # to find: total number of cars in the parking lot

    # solution:
    # calculate the sum of cars originally in the parking lot and the arriving cars
    cars_total = cars_originally + cars_arriving
    return cars_total


FUNCTIONS.append(solution)


def solution():
    # given:
    chocolates_leah_originally = 32  # number of chocolates Leah originally had
    chocolates_sister_originally = 42  # number of chocolates Leah's sister originally had
    chocolates_eaten = 35  # number of chocolates Leah and her sister ate

    # to find: total number of chocolates left

    # solution:
    # first, calculate how many chocolates the sisters originally had in total
    chocolates_total_originally = chocolates_leah_originally + chocolates_sister_originally
    # next, subtract the total number of chocolates they ate
    chocolates_left = chocolates_total_originally - chocolates_eaten
    return chocolates_left


FUNCTIONS.append(solution)


def solution():
    # given:
    lollipops_jason_originally = 20  # number of lollipops Jason had originally
    lollipops_jason_now = 12  # number of lollipops Jason has left

    # to find: number of lollipops Jason gave to Denny

    # solution:
    # calculate the difference between lollipops Jason had originally had lollipops he has now
    lollipops_given_to_denny = lollipops_jason_originally - lollipops_jason_now
    return lollipops_given_to_denny


FUNCTIONS.append(solution)


def solution():
    # given:
    shawn_toys_originally = 5  # number of toys Shawn originally has
    new_toys_from_each_parent = 2  # number of toys his mom and his dad each gave him for Christmas

    # to find: total number of toys Shawn has now

    # solution:
    # first, calculate how many toys in total Shawn got for Christmas
    new_toys_total = 2 * new_toys_from_each_parent
    # next, calculate the sum of the number of toys Shawn originally had and the number of new toys he got
    shawn_toys_now = shawn_toys_originally + new_toys_total
    return shawn_toys_now


FUNCTIONS.append(solution)


def solution():
    # given:
    computers_originally = 9  # number of computers originally in the server room
    computers_installed_each_day = 5  # number of new computers installed each day
    number_of_days = 4  # number of days from monday to thursday

    # to find: total number of computers in the server room now

    # solution:
    # first, calculate how many new computers were installed in total
    computers_installed_total = number_of_days * computers_installed_each_day
    # next, calculate sum of the number of computers originally in the room and the total number of new computers installed
    computers_now = computers_originally + computers_installed_total
    return computers_now


FUNCTIONS.append(solution)


def solution():
    # given:
    golf_balls_originally = 58  # number of golf balls Michael originally had
    golf_balls_lost_tuesday = 23  # number of golf balls lost on tuesday
    golf_balls_lost_wednesday = 2  # number of golf balls lost on wednesday

    # to find: number of golf balls left at the end of wednesday

    # solution:
    # first, calculate the total number of golf balls Michael lost
    golf_balls_lost_total = golf_balls_lost_tuesday + golf_balls_lost_wednesday
    # next, calculate the difference between the original number of golf balls and the number of golf balls lost
    golf_balls_left = golf_balls_originally - golf_balls_lost_total
    return golf_balls_left


FUNCTIONS.append(solution)


def solution():
    # given:
    dollars_originally = 23  # number of dollars Olivia had originally
    bagels = 5  # number of bagels Olivia bought
    price_per_bagel = 3  # number of dollars each bagel cost

    # to find: how many dollars Olivia has left?

    # solution:
    # first, calculate the total cost of the bagels
    bagels_cost_total = bagels * price_per_bagel
    # next, calculate the difference between Olivia's original money and what she paid for the bagels
    dollars_left = dollars_originally - bagels_cost_total
    return dollars_left


FUNCTIONS.append(solution)

