# Example ReasoningFlow splits

## keepdoor_68__step_007__sentence_060

> But row4 column3 is wall, column2 is open, but row4 column2 is open but to go from (3,3) to (4,3) blocked, but could go from (3,3) left to (3,2), down to (4,2) open, then down to (5,2) open, then right to (5,3) open, then right to (5,4) # blocked.

- `reasoning` [0:247]: 'But row4 column3 is wall, column2 is open, but row4 column2 is open but to go from (3,3) to (4,3) blocked, but could go from (3,3) left to (3,2), down to (4,2) open, then down to (5,2) open, then right to (5,3) open, then right to (5,4) # blocked.'

## keepdoor_36__step_008__sentence_165

> But we could go up to (5,1) then right to (5,2) then up to (4,2) then up to (3,2) then up to (2,2) then up to (1,2) then right to (1,3) then right to (1,4) blocked.

- `reasoning` [0:164]: 'But we could go up to (5,1) then right to (5,2) then up to (4,2) then up to (3,2) then up to (2,2) then up to (1,2) then right to (1,3) then right to (1,4) blocked.'

## keepdoor_59__step_002__sentence_071

> Could we go from key to door by staying at (3,2) and then moving right to (3,3), then down to (4,3), then down to (5,3), then down to (6,3), then down to (7,3).

- `reasoning` [0:160]: 'Could we go from key to door by staying at (3,2) and then moving right to (3,3), then down to (4,3), then down to (5,3), then down to (6,3), then down to (7,3).'

## keepdoor_59__step_002__sentence_121

> What about right from (2,2) to (2,3), then down to (3,3), then down to (4,3), then down to (5,3), then right to (5,4) but (5,4) is '#', can't.

- `reasoning` [0:142]: "What about right from (2,2) to (2,3), then down to (3,3), then down to (4,3), then down to (5,3), then right to (5,4) but (5,4) is '#', can't."

## keepdoor_56__step_007__sentence_050

> So maybe go further right to (3,5) then down to (4,5) open, then down to (5,5) open, then right to (5,6) open, then right to (5,7).

- `reasoning` [0:131]: 'So maybe go further right to (3,5) then down to (4,5) open, then down to (5,5) open, then right to (5,6) open, then right to (5,7).'

## keepdoor_33__step_001__sentence_203

> But maybe there is shorter path: After picking key at (3,7), maybe we can go to (5,7) via (4,7) down, then left to (5,6) then left to (5,5) etc. But we counted that path: (4,7)->(5,7) -> (5,6) -> (5,5) -> (5,4) blocked.

- `reflection` [0:219]: 'But maybe there is shorter path: After picking key at (3,7), maybe we can go to (5,7) via (4,7) down, then left to (5,6) then left to (5,5) etc. But we counted that path: (4,7)->(5,7) -> (5,6) -> (5,5) -> (5,4) blocked.'

## keepdoor_92__step_004__sentence_136

> Right? (7,6) then up (6,6) then left (6,5) then left (6,4) then left (6,3) then up (5,3) that's 6.

- `reasoning` [0:98]: "Right? (7,6) then up (6,6) then left (6,5) then left (6,4) then left (6,3) then up (5,3) that's 6."

## keepdoor_30__step_004__sentence_050

> Wait maybe we can go around: from (2,7) go left to (2,6), (2,5), then down to (3,5), (4,5), (5,5), (6,5), (7,5), then left to (7,4), up to (6,4), up to (5,4) but (5,4) is '#' at row5 col4?

- `reasoning` [0:188]: "Wait maybe we can go around: from (2,7) go left to (2,6), (2,5), then down to (3,5), (4,5), (5,5), (6,5), (7,5), then left to (7,4), up to (6,4), up to (5,4) but (5,4) is '#' at row5 col4?"

## keepdoor_77__step_003__sentence_125

> So path to goal: from (3,5) we can go down to (4,5) (now open), then down to (5,5), then right to (5,6), then right to (5,7).

- `reasoning` [0:125]: 'So path to goal: from (3,5) we can go down to (4,5) (now open), then down to (5,5), then right to (5,6), then right to (5,7).'

## keepdoor_80__step_004__sentence_059

> So path: (6,1)->(5,1)->(5,2)->(5,3)->(5,4?) can't because door locked, but we can go to (5,5) maybe via (5,4) but blocked.

- `reasoning` [0:122]: "So path: (6,1)->(5,1)->(5,2)->(5,3)->(5,4?) can't because door locked, but we can go to (5,5) maybe via (5,4) but blocked."
