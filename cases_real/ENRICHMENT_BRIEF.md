# Real-evidence enrichment brief: cases 001, 009, 010

Goal: make the audio and text evidence of the three real cases look like real
evidence (longer recordings, several speakers, people describing each other,
self-identification, metadata) while the **video stays exactly as it is**.
Everything below fits what is on camera.

Enrich toward what real evidence looks like, not toward what the pipeline
handles well. Scores are expected to drop at first: richer, messier evidence
exercises more of the pipeline.

When done, the answer keys (`cases_real/CASE_*_answer_key.json`) are rewritten
from the new evidence **before** any pipeline run.

---

## Rules for all three cases

**Files**
- Keep the folder layout, file naming (`CaseXXX_Audio_ClipN`, `Text/O*.json`)
  and manifest format. New recordings take the next clip number.
- A manifest's `start_time` is when the recording begins; keep it consistent
  with the camera clock.
- Text files keep their fields (`channel`, `author`, `timestamp`, `doc_type`,
  `text`). Headers (From / To / Subject) and signatures go **inside `text`**,
  as they would in a real export.

**Voices**
- One consistent voice per person across all their recordings.
- Clearly different voices for different people: different gender or age
  where plausible, not just a slightly different tone. (Case 009's reworked
  clip 5 at ~140 Hz was still too close to the suspect's 117-133 Hz.)
- Each main speaker gets at least one recording with **8 s or more** of
  their own speech. Keep **one short clip (2-3 s) per case**: real cases
  have them.
- Two-person recordings take turns normally. An officer or dispatcher is a
  new person.

**Content**
- Write it the way people talk: vague times ("around ten"), imprecise
  colours ("dark jacket" for a grey hoodie), fillers, guesses about age and
  gender.
- Do not label roles in the content ("I am the suspect"), and do not make
  every identity link explicit. Covert messages carry no names.
- Keep the existing contradictions; the new ones below are optional extras.

---

## Case 001

**Fixed by the video**
- Suspect: man, pink cap, face mask, dark grey hoodie, black track pants with
  white stripes, white sneakers, black backpack.
  10:00:58 approaches (CAM 01) - 10:03:34 stands outside looking around
  (CAM 03) - 10:04:08 enters (CAM 01) - 10:05:20 exits (CAM 01) - 10:05:50
  exits again (CAM 03).
- Witness: older woman, grey hair, mustard kurta, maroon shawl, green bag,
  watching at 10:05:50 (CAM 03).

**Audio**

| Clip | Source / start | Length / speakers | Content |
|---|---|---|---|
| 1 (keep) | witness_statement, 10:05:39 | 2-3 s, witness | "I was not paying attention to anyone leaving the area." Unchanged: the short clip, and the denial the footage contradicts. |
| 2 (new) | phone_record, ~10:07:10 | 25-40 s, dispatcher + witness (older woman's voice) | Gives her name and location ("My name is Kamala Iyer, I'm at the ATM on ..."). Describes "a young man, pink cap, mask, grey sweatshirt, backpack" coming out and walking off fast; he had been "standing outside looking around for a bit". Gives a call-back number. Contradicts clip 1. |
| 3 (new) | witness_statement, ~10:45 | 45-60 s, officer + witness (same voice as clip 2) | Vague time ("around five past ten"), imprecise colour ("dark jacket"). "I didn't see what he did inside." "It was only him, nobody else with him" - contradicts the email. |

**Text**

| Doc | Channel / time | Content |
|---|---|---|
| O8 (enrich) | email, 10:05:54 | Add *From:* a personal address with no name, *Subject:* "something odd at the ATM". Keep "Two individuals left together quickly. One carrying a bag." Add "I was across the road" - the writer is NOT the woman beside the booth but a third person, not on camera. |
| new | atm_log, 10:04:30 - 10:05:10 | Machine log: "card reader fault", "fascia door sensor open", "no transaction completed". The only evidence of the tampering inside. |

**Findings this sets up:** the double exit; the witness's denial against her
own call and the footage; the email's head-count against the footage and the
witness ("only him"); tampering backed only by the machine log; an email
author who must NOT be merged with the witness.

---

## Case 009

**Fixed by the video (10:02:41 - 10:02:50, CAM 01)**
- Two people leave together: one in a black cap, black hoodie, dark trousers,
  black shoes; the other in a grey long-sleeve shirt, dark trousers, black
  backpack.
- A bystander in a light blue shirt and dark trousers watches from the
  pavement. Footage quality is low.

**Audio**

| Clip | Source / start | Length / speakers | Content |
|---|---|---|---|
| 1 (keep) | phone_record, 10:00:34 | ~2 s, suspect_1 | "I can see the ATM now." The short clip. |
| 2 (enrich) | intercepted_call, 10:00:44 | 10-15 s, suspect_1 + suspect_2 (clearly different voices) | s1: "In the enclosure now, all good." s2: "Road's clear. Be quick, I'm by the tea stall." |
| 3 (enrich) | intercepted_call, 10:01:52 | 8-12 s, suspect_2 | "I've got eyes on the ATM entrance... nobody coming." Settles that the lookout is suspect_2. Source changes from mic_booth. |
| 4 (rework) | interview, next day 2024-01-16 ~11:00 | 45-60 s, officer + suspect_1 | Officer names him ("Rahul Verma") and reads out his phone number: "this number sent these messages". He denies: "I left the area around quarter to ten, long before all this." Contradicts his messages and the footage; links the voice to the texts; tests the long same-name gap. |
| 5 (yours) | phone_record, 10:02:49 | 15-30 s, the bystander (clearly different voice from both suspects) | Gives a name and workplace ("I'm Arjun, I work at the pharmacy next door"). Describes "one in a black cap and hoodie, the other in a grey shirt with a backpack, walking fast towards the bus stop". |

**Text**

| Doc | Change |
|---|---|
| O2, O4, O6, O8, O10 | Keep the terse covert style, timestamps and "(forwarded)" on O4. Add *From:* suspect_1's number, *To:* suspect_2's number. No names. |
| new | sms from suspect_2's number, 10:01:55: "Clear. Go." Gives the second person a text trail. |
| new | statement, ~11:30: the bank guard's written statement, signed with name and phone. "Saw two men walk off quickly, one with a backpack", time "around 10, maybe a bit after", didn't see faces. |

**Findings this sets up:** the denial against his messages and the footage;
the second person and the lookout identified through their calls and number;
the voice linked to the texts via the interview; the witness backed up by the
footage; the guard's vague time.

---

## Case 010

**Fixed by the video (10:03:28 - 10:03:35, CAM 01)**
- A man in a black hoodie or jacket, black trousers and black baseball cap
  comes out of the booth.
- A bystander in a light blue button-down shirt and blue jeans watches him
  walk away. Match the bystander's gender and age to what is seen in the
  clip, not to a guess.

**Audio**

| Clip | Source / start | Length / speakers | Content |
|---|---|---|---|
| 2 (enrich) | phone_record, 10:03:25 | 15-25 s, the bystander, calling someone while watching | "This is Neha, I'm outside the ATM... a guy in a black hoodie and black cap just came out really fast and walked off that way." |
| 1 (enrich) | witness_statement, 10:10:03 | 40-60 s, officer + bystander (same voice as clip 2) | Same name and a phone number. Vague time ("a little after ten"). "He was inside a few minutes." "I didn't see anyone else." |
| new short clip | phone_record, ~10:20 | 2-3 s, dispatcher only | "Unit on its way, stay where you are." The short clip; a different person. |

**Text**

| Doc | Change |
|---|---|
| O1-O5 | Keep wording and timestamps, including "Reader fitted. No one saw." and the early-stamped forwarded "Done here. Moving." Add *From:* the suspect's number, *To:* an unknown number. No names. |
| O11 | Keep it corrupted, exactly as it is. |
| new | atm_log, 10:02:40: "card reader housing tamper sensor triggered" - independent evidence of the tampering. |
| new | technician report, 2024-01-16 ~09:15, signed with name and employee ID: "skimming device found fitted over the card slot", with a description. |

**Optional - an FIR:** there is no FIR in the current evidence (the pipeline
then works from a placeholder). Adding one (see "Optional: an FIR per case"
below) exercises the FIR checks; e.g. an FIR naming two witnesses when the
evidence shows one gives the "FIR vs evidence" finding.

**Findings this sets up:** "No one saw" against the footage and the call; the
forwarded message timed before the act and the camera; the corrupted SMS to
be flagged; tampering backed by the machine log and the technician; the
missing second witness; the witness's two recordings linked by voice and name.

---

## Optional: an FIR per case

None of the three cases has an FIR, so the pipeline's FIR checks (people,
roles, time window, location, items, reporting delay) have nothing to check.
To exercise them, add one FIR text per case (the FIR agent reads it): the
complainant (named, with phone - a self-identification link), the reported
time window and location, the people involved by role, items taken, and when
it was filed. Put one or two realistic discrepancies in it - a time window a
few minutes off, one witness more than the evidence shows - and leave the
rest consistent with the video.

---

## What this exercises

| Fix | Exercised by |
|---|---|
| Voice-print trust by speech duration | short clips beside 8-60 s recordings; distinct voices |
| Appearance and gender | witness descriptions (imprecise) against the video agent's |
| Self-identification links | names, phone numbers, signatures |
| Same name days apart | case 009's next-day interview; case 010's technician |
| Consistency checks | double exit, denials, "No one saw", wrong head-count, early timestamp |
| Multi-speaker diarization | dispatcher and officer recordings |
| Integrity flag | case 010's O11 |
