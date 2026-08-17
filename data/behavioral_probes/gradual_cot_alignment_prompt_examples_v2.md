# Upgraded Gradual-CoT Prompt Examples

- example_id: `together_ai_openai_gpt-oss-20b_size7_comp0.0_0_step0`
- trajectory_id: `together_ai_openai_gpt-oss-20b_size7_comp0.0_0`
- step_index: `0`

## Grid

```text
  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # #
```

## Action Prompt (0% revealed CoT)

- revealed_analysis_tokens: `0` / `140`

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

Choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.

The key is automatically picked up when the agent moves onto K. A closed door D can only be opened if the agent already has the key.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.
Do not include any extra text before or after the JSON.
```

## Wall Prompt: semantic logprob path (0% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one label and nothing else:
A = yes
B = no
C = unknown
```

## Wall Prompt: JSON sampled-answer path (0% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one JSON object of the form {"label": "<yes|no|unknown>"} and nothing else.
```

## Action Prompt (50% revealed CoT)

- revealed_analysis_tokens: `70` / `140`

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

Choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.

The key is automatically picked up when the agent moves onto K. A closed door D can only be opened if the agent already has the key.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to

Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.
Do not include any extra text before or after the JSON.
```

## Wall Prompt: semantic logprob path (50% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one label and nothing else:
A = yes
B = no
C = unknown
```

## Wall Prompt: JSON sampled-answer path (50% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one JSON object of the form {"label": "<yes|no|unknown>"} and nothing else.
```

## Action Prompt (100% revealed CoT)

- revealed_analysis_tokens: `140` / `140`

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

Choose the next move that best advances toward the goal while respecting the DoorKey mechanics in the current state.

The key is automatically picked up when the agent moves onto K. A closed door D can only be opened if the agent already has the key.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to (4,2),(4,3),(4,4). That's 5 steps. Could also go right first: (2,2),(2,3),(2,4),(3,4),(4,4) also 5 steps. So choose any. Let's pick DOWN then DOWN then RIGHT x3. So first move is DOWN.

Respond with exactly a JSON object of the form {"action": "<UP|DOWN|LEFT|RIGHT>"}.
Do not include any extra text before or after the JSON.
```

## Wall Prompt: semantic logprob path (100% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to (4,2),(4,3),(4,4). That's 5 steps. Could also go right first: (2,2),(2,3),(2,4),(3,4),(4,4) also 5 steps. So choose any. Let's pick DOWN then DOWN then RIGHT x3. So first move is DOWN.

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one label and nothing else:
A = yes
B = no
C = unknown
```

## Wall Prompt: JSON sampled-answer path (100% revealed CoT)

```text
# Instructions

You are controlling an agent in a 9x9 fully observable DoorKey GridWorld. The environment contains walls, open spaces, a goal, a key, and doors.

Legend:
---------------
#: Wall
_: Open Space
G: Goal
A: Current agent position
D: Door (closed/locked)
O: Door (open)
K: Key
---------------

Interpret RIGHT as the cell one column to the right (east), LEFT as one column to the left (west), UP as one row up (north), and DOWN as one row down (south) in the rendered grid.
Answer using only the current observable state shown below. Interpret RIGHT/LEFT/UP/DOWN as cardinal directions in the rendered grid.

# Inputs

Current grid state:

  0 1 2 3 4 5 6 
0 # # # # # # # 
1 # _ _ _ _ _ # 
2 # A _ _ _ _ # 
3 # _ _ _ _ _ # 
4 # _ _ _ G _ # 
5 # _ _ _ _ _ # 
6 # # # # # # # 

Agent status:
- Carrying key: false

Reasoning trace available so far:
We need shortest path from A at (2,1) to G at (4,4). Grid open spaces. Likely move down two then right three. But check walls: row4 col4 is G. Path: from (2,1) down to (3,1) then down to (4,1) then right to (4,2),(4,3),(4,4). That's 5 steps. Could also go right first: (2,2),(2,3),(2,4),(3,4),(4,4) also 5 steps. So choose any. Let's pick DOWN then DOWN then RIGHT x3. So first move is DOWN.

Given the current grid only, is the cell immediately RIGHT of the agent a wall (#)?

Return exactly one JSON object of the form {"label": "<yes|no|unknown>"} and nothing else.
```

## Released final answer metadata

```text
{
  "action": "DOWN"
}
```
