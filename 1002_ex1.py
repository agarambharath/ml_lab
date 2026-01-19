data = [
    (["Young", "High", "No", "Fair"], "No"),
    (["Young", "High", "No", "Excellent"], "No"),
    (["Middle", "High", "No", "Fair"], "Yes"),
    (["Old", "Medium", "Yes", "Fair"], "Yes"),
    (["Old", "Low", "Yes", "Fair"], "Yes")
]

n = len(data[0][0])
# Initialize S and G
S = ["Ø"] * n
G = [["?"] * n]

print("Initial Specific Hypothesis (S):", S)
print("Initial General Hypothesis (G):", G)

# Helper functions
def more_general(h1, h2):
    """Returns True if h1 is more general than or equal to h2"""
    for x, y in zip(h1, h2):
        if x != "?" and x != y:
            return False
    return True

def consistent(h, x):
    """Check if hypothesis h is consistent with example x"""
    for i in range(len(h)):
        if h[i] != "?" and h[i] != x[i]:
            return False
    return True

# Candidate Elimination main loop
for x, label in data:
    print("\nTraining Example:", x, "→", label)

    # Positive example
    if label == "Yes":
        # Remove inconsistent hypotheses from G
        G = [g for g in G if consistent(g, x)]

        # Generalize S
        for i in range(n):
            if S[i] == "Ø":
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = "?"

        print("Updated S:", S)
        print("Updated G:", G)

    # Negative example
    else:
        new_G = []
        for g in G:
            if consistent(g, x):
                # Specialize g
                for i in range(n):
                    if g[i] == "?":
                        if S[i] != "?" and S[i] != x[i]:
                            new_h = g.copy()
                            new_h[i] = S[i]
                            new_G.append(new_h)
            else:
                new_G.append(g)

        # Keep only most general hypotheses
        G = []
        for g in new_G:
            if not any(more_general(g2, g) and g2 != g for g2 in new_G):
                G.append(g)

        print("Updated S:", S)
        print("Updated G:", G)

# Final Result
print("\n==============================")
print("Final Specific Hypothesis (S):", S)
print("Final General Hypotheses (G):", G)
print("==============================")