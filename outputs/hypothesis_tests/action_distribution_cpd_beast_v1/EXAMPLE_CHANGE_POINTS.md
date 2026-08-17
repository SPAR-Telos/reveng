# Example Detected Change Points

Examples are selected by the largest posterior-weighted change in the distance from the initial action distribution within each state group.

## together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_15_step_000

- Sentence position 47; posterior change probability 0.942; recommended action DOWN; optimal action: False.
  Text: "We can go from (1,1) down to row 2,3,4,5,6,7 then across to column 7, then up to row 1?"
- Sentence position 62; posterior change probability 0.998; recommended action DOWN; optimal action: False.
  Text: "We can go down to row2 col1, then across to col5, then up to row1 col5, then right to goal."
- Sentence position 67; posterior change probability 0.712; recommended action RIGHT; optimal action: True.
  Text: "But we also need to consider key at (1,3)."

## together_ai_openai_gpt-oss-20b_rooms2_doorkey_keepdoor_88_step_010

- Sentence position 22; posterior change probability 0.999; recommended action DOWN; optimal action: False.
  Text: "Agent at (3,2)."
- Sentence position 68; posterior change probability 0.965; recommended action DOWN; optimal action: False.
  Text: "Row6 col3 is open."
- Sentence position 93; posterior change probability 0.972; recommended action UP; optimal action: True.
  Text: "Let's see: row1 col4 open, so we can go up to row1 from agent?"
- Sentence position 125; posterior change probability 0.901; recommended action UP; optimal action: True.
  Text: "So we need to be adjacent to door to open it, then we can step onto goal."
- Sentence position 262; posterior change probability 0.996; recommended action UP; optimal action: True.
  Text: "From (3,2) to start path: we need to go to (2,2) or (3,3)."

