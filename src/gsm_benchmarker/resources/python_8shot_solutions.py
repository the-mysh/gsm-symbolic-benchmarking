FUNCTIONS = []


def solution():
    # given:
    trees_start = 15  # trees originally in the grove
    trees_end = 21    # trees after planting

    # to find: number of trees planted

    # calculations:
    trees_planted = trees_end - trees_start

    # return:
    return trees_planted


FUNCTIONS.append(solution)


def solution():
    # given:
    cars_start = 3  # cars originally in the lot
    cars_new = 2    # cars that arrived

    # to find: total cars in the lot

    # calculations:
    cars_total = cars_start + cars_new

    # return:
    return cars_total


FUNCTIONS.append(solution)


def solution():
    # given:
    leah_start = 32    # chocolates Leah started with
    sister_start = 42  # chocolates sister started with
    eaten = 35         # total chocolates eaten

    # to find: total chocolates left

    # calculations:
    total_start = leah_start + sister_start
    left = total_start - eaten

    # return:
    return left


FUNCTIONS.append(solution)


def solution():
    # given:
    jason_start = 20  # lollipops Jason started with
    jason_end = 12    # lollipops Jason has now

    # to find: number given to Denny

    # calculations:
    given_away = jason_start - jason_end

    # return:
    return given_away


FUNCTIONS.append(solution)


def solution():
    # given:
    toys_start = 5  # toys Shawn started with
    toys_per_parent = 2  # toys received from each parent

    # to find: total toys Shawn has now

    # calculations:
    toys_received = 2 * toys_per_parent
    toys_end = toys_start + toys_received

    # return:
    return toys_end


FUNCTIONS.append(solution)


def solution():
    # given:
    computers_start = 9  # computers originally in the room
    per_day = 5          # computers installed each day
    days = 4             # total days (monday to thursday)

    # to find: total computers now

    # calculations:
    added_total = days * per_day
    computers_end = computers_start + added_total

    # return:
    return computers_end


FUNCTIONS.append(solution)


def solution():
    # given:
    balls_start = 58  # golf balls Michael started with
    lost_tues = 23    # lost on Tuesday
    lost_wed = 2      # lost on Wednesday

    # to find: golf balls left

    # calculations:
    lost_total = lost_tues + lost_wed
    balls_left = balls_start - lost_total

    # return:
    return balls_left


FUNCTIONS.append(solution)


def solution():
    # given:
    money_start = 23  # money Olivia started with
    count = 5         # number of bagels
    cost_each = 3     # price per bagel

    # to find: money left

    # calculations:
    total_cost = count * cost_each
    money_left = money_start - total_cost

    # return:
    return money_left


FUNCTIONS.append(solution)

