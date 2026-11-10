WIDTH, HEIGHT = 1920, 1080
JOYSTICK = (180, 900)
ATTACK = (1723, 793)
SUPER = (1512, 881)
GADGET = (1645, 989)
HYPERCHARGE = (1400, 990)
PROCEED = (1660, 980)
PLAY_AGAIN = (1300, 990)
CONTINUE = (700, 1000)
BACK = (100, 60)
CENTER = (960, 540)
EXIT_SHOWDOWN = (950, 1000)
EXIT_FRIEND_MENU = (1830, 50)

MOVEMENT_POINTER = 1
ACTION_POINTER = 2


def validate_resolution(width: int, height: int):
    if width <= 0 or height <= 0 or abs(width / height - WIDTH / HEIGHT) > 0.03:
        raise ValueError("Use a landscape 16:9 device resolution (for example 1280x720)")
