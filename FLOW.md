# SDK Flow: What the Program Does

A Duckpond Next session, action by action.

## Startup

1. Obtain the identity documents (soul, bill of rights, etc.)
2. Obtain environmental context (client name, weather, hostname)
3. Obtain yesterday's capsules (what happened yesterday, what happened last night)
4. Obtain the letter from last night
5. Assemble all of the above into the system prompt
6. Start the main claude engine with that system prompt

## First Turn

7. Receive a message from a human
8. Obtain today-so-far summary
9. Obtain context files (ALPHA.md files and hints)
10. Obtain calendar events
11. Obtain todos
12. Recall relevant memories based on the human's message
13. Check for memorables from Intro (none exist yet on first turn)
14. Check context usage level
15. Get current timestamp
16. Assemble all of 8-15 plus the human's message into one user message
17. Send that message to the engine
18. Receive the engine's response as a stream of events
19. Deliver those events to the consumer
20. Archive the turn
21. Broadcast the turn to any connected watchers
22. Extract memorables from the turn for next time

## Subsequent Turns

23. Receive a message from a human (or from Frobozz, or from wherever)
24. Recall relevant memories based on that message
25. Check for memorables from Intro (from step 22 of the previous turn)
26. Check context usage level
27. Get current timestamp
28. Assemble 24-27 plus the message into one user message
29. Send that message to the engine
30. Receive the engine's response as a stream of events
31. Deliver those events to the consumer
32. Archive the turn
33. Broadcast the turn
34. Extract memorables for next time

## Compaction

35. Compaction triggers (human command or context threshold)
36. Intercept the compaction to inject identity so the summary sounds right
37. New context window begins — next message goes back to step 8

## Shutdown

38. Stop the engine
39. Clean up
