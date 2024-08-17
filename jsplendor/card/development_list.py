# price array - [(w)hite, bl(u)e, (g)reen, (r)ed, blac(k)]
# price array - [w, u, g, r, k]

# name / level / VP / gem_color / price 
l1_list = [
    # level1 black
    ('l1_black_1', 1, 0, 'black', [1, 1, 1, 1, 0]),
    ('l1_black_2', 1, 0, 'black', [1, 2, 1, 1, 0]),
    ('l1_black_3', 1, 0, 'black', [2, 2, 0, 1, 0]),
    ('l1_black_4', 1, 0, 'black', [0, 0, 1, 3, 1]),
    ('l1_black_5', 1, 0, 'black', [0, 0, 2, 1, 0]),
    ('l1_black_6', 1, 0, 'black', [2, 0, 2, 0, 0]),
    ('l1_black_7', 1, 0, 'black', [0, 0, 3, 0, 0]),
    ('l1_black_8', 1, 1, 'black', [0, 4, 0, 0, 0]),

    # level1 blue
    ('l1_blue_1', 1, 0, 'blue', [1, 0, 1, 1, 1]),
    ('l1_blue_2', 1, 0, 'blue', [1, 0, 1, 2, 1]),
    ('l1_blue_3', 1, 0, 'blue', [1, 0, 2, 2, 0]),
    ('l1_blue_4', 1, 0, 'blue', [0, 1, 3, 1, 0]),
    ('l1_blue_5', 1, 0, 'blue', [1, 0, 0, 0, 2]),
    ('l1_blue_6', 1, 0, 'blue', [0, 0, 2, 0, 2]),
    ('l1_blue_7', 1, 0, 'blue', [0, 0, 0, 0, 3]),
    ('l1_blue_8', 1, 1, 'blue', [0, 0, 0, 4, 0]),

    # level1 white
    ('l1_white_1', 1, 0, 'white', [0, 1, 1, 1, 1]),
    ('l1_white_2', 1, 0, 'white', [0, 1, 2, 1, 1]),
    ('l1_white_3', 1, 0, 'white', [0, 2, 2, 0, 1]),
    ('l1_white_4', 1, 0, 'white', [3, 1, 0, 0, 1]),
    ('l1_white_5', 1, 0, 'white', [0, 0, 0, 2, 1]),
    ('l1_white_6', 1, 0, 'white', [0, 2, 0, 0, 2]),
    ('l1_white_7', 1, 0, 'white', [0, 3, 0, 0, 0]),
    ('l1_white_8', 1, 1, 'white', [0, 0, 4, 0, 0]),

    # level1 green
    ('l1_green_1', 1, 0, 'green', [1, 1, 0, 1, 1]),
    ('l1_green_2', 1, 0, 'green', [1, 1, 0, 1, 2]),
    ('l1_green_3', 1, 0, 'green', [0, 1, 0, 2, 2]),
    ('l1_green_4', 1, 0, 'green', [1, 3, 1, 0, 0]),
    ('l1_green_5', 1, 0, 'green', [2, 1, 0, 0, 0]),
    ('l1_green_6', 1, 0, 'green', [0, 2, 0, 2, 0]),
    ('l1_green_7', 1, 0, 'green', [0, 0, 0, 3, 0]),
    ('l1_green_8', 1, 1, 'green', [0, 0, 0, 0, 4]),

    # level1 red
    ('l1_red_1', 1, 0, 'red', [1, 1, 1, 0, 1]),
    ('l1_red_2', 1, 0, 'red', [2, 1, 1, 0, 1]),
    ('l1_red_3', 1, 0, 'red', [2, 0, 1, 0, 2]),
    ('l1_red_4', 1, 0, 'red', [1, 0, 0, 1, 3]),
    ('l1_red_5', 1, 0, 'red', [0, 2, 1, 0, 0]),
    ('l1_red_6', 1, 0, 'red', [2, 0, 0, 2, 0]),
    ('l1_red_7', 1, 0, 'red', [3, 0, 0, 0, 0]),
    ('l1_red_8', 1, 1, 'red', [4, 0, 0, 0, 0]),
]

l2_list = [
    # level2 black
    ('l2_black_1', 2, 1, 'black', [3, 2, 2, 0, 0]),
    ('l2_black_2', 2, 1, 'black', [3, 0, 3, 0, 2]),
    ('l2_black_3', 2, 2, 'black', [0, 1, 4, 2, 0]),
    ('l2_black_4', 2, 2, 'black', [0, 0, 5, 3, 0]),
    ('l2_black_5', 2, 2, 'black', [5, 0, 0, 0, 0]),
    ('l2_black_6', 2, 3, 'black', [0, 0, 0, 0, 6]),

    # level2 blue
    ('l2_blue_1', 2, 1, 'blue', [0, 2, 2, 3, 0]),
    ('l2_blue_2', 2, 1, 'blue', [0, 2, 3, 0, 3]),
    ('l2_blue_3', 2, 2, 'blue', [5, 3, 0, 0, 0]),
    ('l2_blue_4', 2, 2, 'blue', [2, 0, 0, 1, 4]),
    ('l2_blue_5', 2, 2, 'blue', [0, 5, 0, 0, 0]),
    ('l2_blue_6', 2, 3, 'blue', [0, 6, 0, 0, 0]),

    # level2 white
    ('l2_white_1', 2, 1, 'white', [0, 0, 3, 2, 2]),
    ('l2_white_2', 2, 1, 'white', [2, 3, 0, 3, 0]),
    ('l2_white_3', 2, 2, 'white', [0, 0, 1, 4, 2]),
    ('l2_white_4', 2, 2, 'white', [0, 0, 0, 5, 3]),
    ('l2_white_5', 2, 2, 'white', [0, 0, 0, 5, 0]),
    ('l2_white_6', 2, 3, 'white', [6, 0, 0, 0, 0]),

    # level2 green
    ('l2_green_1', 2, 1, 'green', [3, 0, 2, 3, 0]),
    ('l2_green_2', 2, 1, 'green', [2, 3, 0, 0, 2]),
    ('l2_green_3', 2, 2, 'green', [4, 2, 0, 0, 1]),
    ('l2_green_4', 2, 2, 'green', [0, 5, 3, 0, 0]),
    ('l2_green_5', 2, 2, 'green', [0, 0, 5, 0, 0]),
    ('l2_green_6', 2, 3, 'green', [0, 0, 6, 0, 0]),

    # level2 red
    ('l2_red_1', 2, 1, 'red', [2, 0, 0, 2, 3]),
    ('l2_red_2', 2, 1, 'red', [0, 3, 0, 2, 3]),
    ('l2_red_3', 2, 2, 'red', [1, 4, 2, 0, 0]),
    ('l2_red_4', 2, 2, 'red', [3, 0, 0, 0, 5]),
    ('l2_red_5', 2, 2, 'red', [0, 0, 0, 0, 5]),
    ('l2_red_6', 2, 3, 'red', [0, 0, 0, 6, 0]),
]

l3_list = [
    # level3 black
    ('l3_black_1', 3, 3, 'black', [3, 3, 5, 3, 0]),
    ('l3_black_2', 3, 4, 'black', [0, 0, 0, 7, 0]),
    ('l3_black_3', 3, 4, 'black', [0, 0, 3, 6, 3]),
    ('l3_black_4', 3, 5, 'black', [0, 0, 0, 7, 3]),

    # level3 blue
    ('l3_blue_1', 3, 3, 'blue', [3, 0, 3, 3, 5]),
    ('l3_blue_2', 3, 4, 'blue', [7, 0, 0, 0, 0]),
    ('l3_blue_3', 3, 4, 'blue', [6, 3, 0, 0, 3]),
    ('l3_blue_4', 3, 5, 'blue', [7, 3, 0, 0, 0]),

    # level3 white
    ('l3_white_1', 3, 3, 'white', [0, 3, 3, 5, 3]),
    ('l3_white_2', 3, 4, 'white', [0, 0, 0, 0, 7]),
    ('l3_white_3', 3, 4, 'white', [3, 0, 0, 3, 6]),
    ('l3_white_4', 3, 5, 'white', [3, 0, 0, 0, 7]),

    # level3 green
    ('l3_green_1', 3, 3, 'green', [5, 3, 0, 3, 3]),
    ('l3_green_2', 3, 4, 'green', [0, 7, 0, 0, 0]),
    ('l3_green_3', 3, 4, 'green', [3, 6, 3, 0, 0]),
    ('l3_green_4', 3, 5, 'green', [0, 7, 3, 0, 0]),

    # level3 red
    ('l3_red_1', 3, 3, 'red', [3, 5, 3, 0, 3]),
    ('l3_red_2', 3, 4, 'red', [0, 0, 7, 0, 0]),
    ('l3_red_3', 3, 4, 'red', [0, 3, 6, 3, 0]),
    ('l3_red_4', 3, 5, 'red', [0, 0, 7, 3, 0]),
]






