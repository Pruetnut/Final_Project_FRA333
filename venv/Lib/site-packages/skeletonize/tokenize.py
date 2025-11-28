def categorize_char(char, last_was_slash):
    if last_was_slash:
        return None
    if char.isspace():
        return None
    elif char.isalnum():
        return True
    else:
        return None


def tokenize(code):
    last_cat = None
    last_was_slash = False
    result = []
    for char in code:
        cat = categorize_char(char, last_was_slash)
        if last_cat is not None and last_cat == cat:
            result[-1] += char
        else:
            result.append(char)
        last_was_slash = char == "\\"
        last_cat = cat
    return result
