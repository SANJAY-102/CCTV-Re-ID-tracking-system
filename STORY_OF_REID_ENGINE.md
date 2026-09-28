# Teaching a Browser to Recognize Humans: Our Journey Through AI Chaos

### By Gyaini Engineering

When we started this project, we set ourselves a stubborn challenge: teach a standard web browser to recognize and track people on office CCTV cameras—without sending private video to the cloud and without using expensive supercomputers. Everything had to run locally on a regular laptop, even though computer vision typically demands heavy cloud servers.

The early days were humbling. It was a comedy of errors.

Our first lesson was that Artificial Intelligence can be surprisingly foolish. In glossy demos, AI looks flawless. In our real office footage, it was chaotic. When we adjusted sensitivity to catch distant figures, the AI began hallucinating. It spotted an empty black swivel chair and declared it a human employee. A laptop bag on a table was given an employee ID. For two full days, our system diligently tracked office furniture.

Then came the movement problems. While stationary people were easy to spot, the moment someone stood up and started walking, the AI failed to detect the moving person altogether. Their tracking box would flicker and vanish mid-stride, losing them completely just because they walked across the room.

Even worse, whenever two people came near each other—whether stopping to chat or passing by—their employee IDs instantly switched. The moment their paths brushed together for a fraction of a second, the AI got hopelessly confused. Person A walked away carrying Person B's ID, and Person B was suddenly tracked as Person A.

Things got weirder when people sat down. Because CCTV cameras look down from high ceilings, a standing person looks tall and slim. But when that person sits in an office chair, they shrink into a small cluster from above. To the AI, their shape changed so drastically that it assumed the seated person had vanished into thin air, and a stranger had mysteriously materialized in their chair.

To make matters worse, regular browsers were never built for heavy AI tasks. Whenever the computer broke a sweat, the video skipped ahead, leaving the AI behind and dropping people.

We tore down the software and rebuilt it twelve separate times. 

That was when we had our breakthrough: **AI is not a magical brain; it is just a short-sighted sensor that needs common sense.**

Instead of letting the AI guess blindly, we wrapped everyday physical rules around it:
1. **No ID switching:** We tracked direction and speed. When people came near each other, the math strictly forbade their IDs from swapping.
2. **Never losing a moving person:** We added motion prediction so fast-moving employees were never dropped mid-stride.
3. **Remembering postures:** We gave each person a memory bank that remembers what they look like both standing up and sitting down.
4. **Saving a seat:** If someone stepped away, the system kept a "ghost memory" of their desk, welcoming them right back when they returned.
5. **Patience over speed:** We forced the video to inspect every frame carefully rather than rushing and making sloppy mistakes.

The day we watched six people walk, sit, cross paths, step away, and return—with every badge staying locked on the right person without a single glitch—we knew we had won.

Sometimes, making AI smarter doesn't mean buying bigger models. It just means giving it good old-fashioned common sense.
