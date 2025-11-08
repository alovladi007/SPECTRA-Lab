DISALLOWED = [
    "detailed instructions for creating explosives",
    "requests for illegal access or malware creation",
    "medical diagnosis or treatments instructions",
]

def should_refuse(prompt:str)->bool:
    p = prompt.lower()
    return any(k in p for k in ["make a bomb", "zero-day exploit", "crack this account", "prescribe a drug"])

REFUSAL = "I can’t help with that. If you’re working on a legitimate research or safety case, consider high-level discussion or consult an authority."

