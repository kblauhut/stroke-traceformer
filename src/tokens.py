CURVE_PAD_TOKEN = "<CURVE_PAD>"
CURVE_END_TOKEN = "<CURVE_END>"
CURVE_MOVE_TOKEN = "<MOVE>"
CURVE_BEZIER_TOKEN = "<BEZIER>"

def pad_sequence(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value] * max(0, max_len - len(seq))

GRID_SIZE = 100

class CurveTokenizer():
    def __init__(self):
        base_curve_token_list = [CURVE_PAD_TOKEN, CURVE_END_TOKEN, CURVE_MOVE_TOKEN, CURVE_BEZIER_TOKEN]
        curve_token_to_idx = {token: idx for idx, token in enumerate(base_curve_token_list)}
        self.curve_token_to_idx = curve_token_to_idx
        self.curve_idx_to_token = {idx: token for token, idx in curve_token_to_idx.items()}
        self.CURVE_PAD_TOKEN = CURVE_PAD_TOKEN
        self.CURVE_END_TOKEN = CURVE_END_TOKEN
        self.CURVE_MOVE_TOKEN = CURVE_MOVE_TOKEN
        self.CURVE_BEZIER_TOKEN = CURVE_BEZIER_TOKEN
        self.token_idx = len(base_curve_token_list)

        for x in range(-GRID_SIZE, GRID_SIZE+1):
            for y in range(-GRID_SIZE, GRID_SIZE+1):
                string_representation = f"<{x}_{y}>"
                curve_token_to_idx[string_representation] = self.token_idx
                self.curve_idx_to_token[self.token_idx] = string_representation
                self.token_idx += 1

    def get_token_count(self):
        return len(self.curve_token_to_idx)

    def get_token_idx(self, token):
        return self.curve_token_to_idx[token]

    def get_idx_token(self, idx):
        for token, token_idx in self.curve_token_to_idx.items():
            if token_idx == idx:
                return token
        return "ERROR"

    def get_coordinate_token(self, x, y):
        string_representation = f"<{round(x)}_{round(y)}>"
        return self.curve_token_to_idx[string_representation]

    def get_coordinate_tokens(self, x, y):
        xDir = 1 if x > 0 else -1
        yDir = 1 if y > 0 else -1
        x = abs(x)
        y = abs(y)
        tokens = []
        while x > GRID_SIZE or y > GRID_SIZE:
            if x > GRID_SIZE and y > GRID_SIZE:
                tokens.append(self.get_coordinate_token(xDir * GRID_SIZE, yDir * GRID_SIZE))
                x -= GRID_SIZE
                y -= GRID_SIZE
            elif x > GRID_SIZE:
                tokens.append(self.get_coordinate_token(xDir * GRID_SIZE, 0))
                x -= GRID_SIZE
            elif y > GRID_SIZE:
                tokens.append(self.get_coordinate_token(0, yDir * GRID_SIZE))
                y -= GRID_SIZE
        if (x != 0 or y != 0):
            tokens.append(self.get_coordinate_token(xDir * x, yDir * y))
        return tokens


    def tokenize_commands(self, commands):
        x = 0
        y = 0
        command_tokens = []
        idx = 0
        for command in commands:
            if (command[0] == 'M'):
                command_tokens.append(
                    self.get_token_idx(self.CURVE_MOVE_TOKEN))

                new_x = command[1]
                new_y = command[2]
                x_diff = new_x - x
                y_diff = new_y - y

                # The first move command should always be at x=0
                if idx == 0:
                    x_diff = 0

                x = new_x
                y = new_y
                command_tokens += self.get_coordinate_tokens(x_diff, y_diff)
            if (command[0] == 'C'):
                command_tokens.append(self.get_token_idx(self.CURVE_BEZIER_TOKEN))
                cx1_diff = command[1] - x
                cy1_diff = command[2] - y
                cx2_diff = command[3] - x
                cy2_diff = command[4] - y
                new_x = command[5]
                new_y = command[6]
                x_diff = new_x - x
                y_diff = new_y - y
                x = new_x
                y = new_y
                command_tokens.append(self.get_coordinate_token(cx1_diff, cy1_diff))
                command_tokens.append(self.get_coordinate_token(cx2_diff, cy2_diff))
                command_tokens.append(self.get_coordinate_token(x_diff, y_diff))
            idx += 1
        command_tokens.append(self.get_token_idx(self.CURVE_END_TOKEN))
        return command_tokens

    def tokenize_commands_pad(self, commands, lenght):
        command_tokens = self.tokenize_commands(commands)
        command_tokens = pad_sequence(command_tokens, lenght, self.get_token_idx(self.CURVE_PAD_TOKEN))
        return command_tokens
