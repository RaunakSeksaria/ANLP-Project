# Audit of `event_ids` in saved positive pairs

This file only inspects metadata, the original articles in the source pairing file, and saved triplet IDs. It does not recompute embeddings or triplet metrics.

**Memberships:** 20,000 across 11,855 articles and 10,000 source story IDs. 4,009 articles have multiple IDs.

**IDs per article:** min 1, median 1, mean 1.6871, max 32.

**Articles per ID:** min/median/mean/max = 2/2/2/2. All 10,000 IDs have one left and one right article.

The IDs represent 8,349 distinct article pairings; repeated pairings can carry several source story IDs.

**Source trace:** 20,000 of 20,000 exact-body/source-story memberships match metadata. Two occurrences have a title variant with the same exact article body; titles are not the membership key.

## Manual interpretation

An ID certifies a pairing in the source file. It does not certify that both stories cover one narrowly defined news event. Judgments here are conservative: `same_specific_story` includes reports and reactions to one identifiable occurrence or announcement; `related_developments` covers distinct developments in a common story arc; `different_events` names separate occurrences; `topic_only_or_insufficient` lacks enough evidence of a common event.

Judgments in the seed-42 sample: related_developments: 5, same_specific_story: 11, different_events: 3, topic_only_or_insufficient: 1.

### triplet_000170 — related_developments

Shared source ID(s): 463590, 760927.

- Anchor **right_002579 (right)** — Donald Trump still against assault weapon ban
  - Beginning: Should Trump keep Pence on the ticket in 2020? President Trump remains opposed to an assault weapon ban as he considers steps to prevent gun violence in schools, the White House said Thursday. White House deputy press secretary Ra…
- Positive **left_000903 (left)** — Donald Trump Argues He Never Said 'Give Teachers Guns' Before Pushing To Give Some Teachers Guns
  - Beginning: President Donald Trump elaborated on his proposal to arm 20 percent of teaching staff at schools in a series of tweets Thursday. Trump first made the proposal during a listening session with students and parents affected by mass s…

Assessment: Same Parkland aftermath, but distinct Trump gun-policy responses: assault-weapon ban versus arming teachers.

### triplet_000176 — same_specific_story

Shared source ID(s): 350130.

- Anchor **left_002077 (left)** — Teresa Heinz Kerry ‘doing better,’ says a teary John Kerry (video)
  - Beginning: Secretary of State John Kerry got a little emotional Wednesday while updating an audience on the condition of his sick wife. “Teresa is doing better under evaluation and we hope improving,” he said as he opened an annual summit be…
- Positive **right_001185 (right)** — John Kerry in tears: 'Teresa is doing better'
  - Beginning: John Kerry in tears: ‘Teresa is doing better’ U.S. Secretary of State John Kerry gave a teary-eyed thank you on Wednesday to well-wishers concerned about his wife Teresa’s medical condition. “Teresa is doing better, under evaluati…

Assessment: John Kerry's tearful public update on Teresa Heinz Kerry's hospitalization.

### triplet_000187 — different_events

Shared source ID(s): 940984.

- Anchor **right_004079 (right)** — FNC's Wallace Presses Pompeo on Why Trump Would Meet with Taliban Days Before 9/11
  - Beginning: On this week’s broadcast of “Fox News Sunday,” anchor Chris Wallace pressed Secretary of State Mike Pompeo on why President Donald Trump planned to meet with Taliban leaders in the United States three days before the anniversary o…
- Positive **left_003767 (left)** — U.S. President Trump Could Meet With Iran's Rouhani at U.N. With No Preconditions: Pompeo
  - Beginning: WASHINGTON — U.S. Secretary of State Mike Pompeo said on Tuesday that President Donald Trump could meet with Iranian President Hassan Rouhani at the upcoming United Nations meeting, with "no preconditions." Pompeo made the comment…

Assessment: Pompeo interview about a Trump–Taliban meeting versus comments about a possible Trump–Rouhani meeting with Iran.

### triplet_000255 — same_specific_story

Shared source ID(s): 85388.

- Anchor **left_004684 (left)** — Former Chicago Mayor Richard M. Daley hospitalized
  - Beginning: Former Chicago Mayor Richard M. Daley remained hospitalized early Sunday morning, two days after he fell ill. Daley, 71, was taken by ambulance to Northwestern Memorial Hospital in Chicago on Friday evening after he returned from …
- Positive **right_005104 (right)** — Former Chicago Mayor Richard Daley likely will remain in ICU for several days
  - Beginning: CHICAGO – Former Chicago Mayor Richard M. Daley likely will remain hospitalized in an intensive care unit for several days as doctors conduct tests to determine why he was feeling ill and disoriented during a conference in Arizona…

Assessment: Richard M. Daley's hospitalization and ICU stay.

### triplet_000400 — same_specific_story

Shared source ID(s): 9674.

- Anchor **left_005667 (left)** — POTUS Dumps Wingnut Judicial Nominee Boggs
  - Beginning: From thinkprogress.org: In a victory for civil rights, abortion rights, and LGBT equality groups, President Obama will not re-nominate former Georgia legislator Michael Boggs to a lifetime position on the federal bench. Boggs, who…
- Positive **right_006122 (right)** — Obama will not re-nominate Michael Boggs for federal judgeship
  - Beginning: President Obama will not re-nominate Georgia judge Michael Boggs to the federal bench, appeasing Democrats who slammed the judicial nominee's position on various social issues. “It is with regret that we announce that the presiden…

Assessment: Obama's decision not to re-nominate Michael Boggs to the federal bench.

### triplet_000859 — different_events

Shared source ID(s): 106273.

- Anchor **right_005304 (right)** — George W. Bush praises Jimmy Kimmel's Oscars opening monologue
  - Beginning: President Donald Trump may not have appreciated Jimmy Kimmel's opening monologue at the 89th Academy Awards, but the late night host has found a fan in former president George W. Bush. The 43rd President of the United States told …
- Positive **left_004878 (left)** — Ellen Asks George W. Bush About His Inauguration Poncho: 'Had You Put One On?'
  - Beginning: Former President George W. Bush appeared on “Ellen” yesterday to talk politics, friendship and the poncho. “The poncho was a problem,” host Ellen DeGeneres began. In case you forgot, the poncho in question was the infamous one Bus…

Assessment: George W. Bush on Jimmy Kimmel versus a separate appearance on Ellen about his inauguration poncho.

### triplet_000870 — related_developments

Shared source ID(s): 606286.

- Anchor **right_003108 (right)** — Trump: New Steps Next Week to 'Keep Country Safe'
  - Beginning: Friday at the White House, during a joint press conference with Japanese Prime Minister Shinzo Abe, when asked about the 9th Circuit Court ruling upholding the blocking of his executive order banning immigrants from seven Muslim-m…
- Positive **left_000026 (left)** — Trump ditches press pool to play golf with Japanese PM
  - Beginning: More disdain for transparency and tradition from the Trump administration. Members of President Trump's press pool on Saturday said they were kept from tagging along as the president golfed with Japanese Prime Minister Shinzo Abe.…

Assessment: Trump–Abe visit links the articles, but one reports an immigration-order press conference and the other a later golf/press-pool incident.

### triplet_000900 — same_specific_story

Shared source ID(s): 377954.

- Anchor **right_002223 (right)** — Obama to daughters: 'Of all that I have done ... I'm most proud to be your dad'
  - Beginning: A visibly emotional President Barack Obama thanked his wife and daughters during his farewell address in Chicago Tuesday evening. Addressing 18-year-old Malia and 15-year-old Sasha, the president said, "You have become two amazing…
- Positive **left_000831 (left)** — Opinion | To President Obama: ‘We Will Miss You’
  - Beginning: To President Obama: ‘We Will Miss You’ To the Editor: Re “Obama’s Farewell to the Nation: ‘Yes, We Did!’” (front page, Jan. 11): On Tuesday night it sunk in that President Obama will be stepping down in a bit more than a week! Pre…

Assessment: Obama's Chicago farewell address; one report and one response letter.

### triplet_001024 — same_specific_story

Shared source ID(s): 854807.

- Anchor **right_006012 (right)** — Obama: 'I had a bad night' in debate
  - Beginning: President Obama said Wednesday he did not hand the election to Republican Mitt Romney with his poor debate performance last week. “Governor Romney had a good night. I had a bad night,” Mr. Obama told ABC News’ Diane Sawyer. “It’s …
- Positive **left_005571 (left)** — Obama Diane Sawyer Interview: President Says Mitt Romney Trying To 'Cloud' Abortion Stance
  - Beginning: WASHINGTON -- President Barack Obama says Mitt Romney's recent remark about not having abortion on his agenda is an attempt "to cloud" his rival's position on the issue. Obama says he had a "bad night" in last week's debate with R…

Assessment: Different answers in the same Obama–Diane Sawyer interview after the Romney debate.

### triplet_001048 — related_developments

Shared source ID(s): 378184.

- Anchor **left_005235 (left)** — This map shows the 71 House Democrats who have said they won't attend Trump's inauguration ceremony
  - Beginning: So far, 71 of the 194 Democratic members of the House have announced they will not attend Donald Trump’s inauguration ceremony on Friday, while the remaining 123 are either attending or have not made their intentions clear. As sho…
- Positive **right_005661 (right)** — Donald Trump: John Lewis Lied About Not Skipping Inaugurations Before Mine
  - Beginning: President-elect Donald Trump continued his feud with Democratic Congressman John Lewis after the civil rights legend said he does not believe Trump is a “legitimate president” and that this is the first inauguration he will not at…

Assessment: Trump inauguration boycott context, but a broad map of boycotting House Democrats versus Trump reacting specifically to John Lewis.

### triplet_001298 — same_specific_story

Shared source ID(s): 918633.

- Anchor **right_005175 (right)** — On last day in Israel, Pence visits Western Wall, Yad Vashem
  - Beginning: JERUSALEM – Vice President Mike Pence is making a visit to the Western Wall and meeting with Israeli President Reuvin Rivlin on his final day of a trip to Israel. The vice president is also to visit the Yad Vashem Holocaust memori…
- Positive **left_004754 (left)** — Pence Visit Was a Balm to Israelis. Now Back to Reality.
  - Beginning: JERUSALEM — Prime Minister Benjamin Netanyahu of Israel positively basked in the glow of Vice President Mike Pence’s visit here, though it may yet turn out to be the halcyon days of a drawn-out, gloomy winter. Sandwiched between M…

Assessment: Pence's Israel visit, including the same Tuesday Western Wall visit; one straight report and one retrospective.

### triplet_001386 — related_developments

Shared source ID(s): 31577, 655396.

- Anchor **right_000566 (right)** — E. Jean Carroll on not being Donald Trump's 'type': 'I'm so glad'
  - Beginning: Is 'gun control' a winning issue for Democrats? Reacting to a recent statement from President Trump, E. Jean Carroll, the writer who accused the president of raping her in the 1990s, said she was “so glad” to find out she was not …
- Positive **left_001933 (left)** — Opinion | When Women Testify to Assault
  - Beginning: To the Editor: Re “Is Trump a Rapist?” (column, nytimes.com, June 26): Regarding E. Jean Carroll’s allegation of sexual assault against Donald Trump, Charles M. Blow writes: “Carroll’s allegation fits the behaviors that have been …

Assessment: Same E. Jean Carroll allegation, but a reaction to a specific Trump remark versus a broader opinion letter about testimony.

### triplet_001431 — same_specific_story

Shared source ID(s): 391482.

- Anchor **left_000123 (left)** — The 18th School Shooting of 2018: Mass Shooter in Parkland, Florida High School Arrested.
  - Beginning: There have been 18 school shootings so far this year, including one at a public school in Florida Wednesday afternoon. The title was changed to include all school shootings of 2018. The 18th school shooting of 2018 remains an acti…
- Positive **right_002294 (right)** — Several hurt in US school shooting: officials, media
  - Beginning: Miami (AFP) – Police were responding to an apparent shooting on Wednesday at a high school in Florida that reportedly left several people injured, officials and local media said. The incident was unfolding at Marjory Stoneman Doug…

Assessment: The shooting at Marjory Stoneman Douglas High School in Parkland.

### triplet_001466 — different_events

Shared source ID(s): 473155.

- Anchor **right_002608 (right)** — Lawyer: Claim that Sean Spicer used racial slur is false
  - Beginning: NEW YORK – A lawyer for Sean Spicer says an accusation that the former White House press secretary used a racial slur to describe a black student while in high school is false. Attorney Michael Bowe says in emails to The Associate…
- Positive **left_002606 (left)** — Sean Spicer Heckled At Book Tour Event In New York City, Called 'Piece Of Garbage'
  - Beginning: Former White House press secretary Sean Spicer was heckled on Wednesday while promoting his new book in New York City. In a video sent to Think Progress reporter Aaron Rupar, an unidentified man can be heard yelling at Spicer duri…

Assessment: Separate Sean Spicer book-tour incidents: a Rhode Island racial-slur accusation versus heckling in New York City.

### triplet_001517 — same_specific_story

Shared source ID(s): 963354.

- Anchor **left_004787 (left)** — Margaret Thatcher And Ronald Reagan: A Special Relationship In Photos
  - Beginning: Former British Prime Minister Margaret Thatcher died Monday at the age of 87, leaving behind -- among many things -- memories of one of the more notable special relationships between leaders of the United Kingdom and United States…
- Positive **right_005211 (right)** — U.S. House of Representatives to Honor Margaret Thatcher
  - Beginning: The U.S. House Of Representatives will pass a resolution this evening by unanimous consent honoring former Prime Minister of the United Kingdom Margaret Thatcher. Lady Thatcher passed away at the age of 87 on Monday in London. The…

Assessment: Margaret Thatcher's death and reactions to it, including a U.S. House tribute.

### triplet_001533 — related_developments

Shared source ID(s): 933945.

- Anchor **right_005188 (right)** — Trump campaign unveils ‘Pencil-neck Adam Schiff’ T-shirts for sale
  - Beginning: President Trump is taking his displeasure with U.S. Rep. Adam Schiff, D-Calif., a step further with the sale of “Pencil-Neck Adam Schiff” T-shirts. The term stems from a Trump campaign rally held Thursday in Grand Rapids, Mich., w…
- Positive **left_004766 (left)** — President Toadstool
  - Beginning: Provoked by Trump’s insulting name for Adam Schiff …. There’s only one person who shut up Donald Trump with his insulting names: Stormy Daniels. After he called her an insulting name, she came back and referred to his “little toad…

Assessment: The 'pencil-neck Adam Schiff' insult connects them; merchandise announcement versus a broader opinion piece on Trump's insults.

### triplet_001568 — topic_only_or_insufficient

Shared source ID(s): 189394.

- Anchor **left_001033 (left)** — Bruce Bartlett: Inside The Cain Tax Plan
  - Beginning: The New York Times: With recent polls showing increased support for Herman Cain as the G.O.P. presidential nominee, attention is being drawn to his platform, especially what he calls the 9-9-9 tax plan. News reports describe it as…
- Positive **right_001021 (right)** — GOP Debate - Romney: Obama Has Divided America
  - Beginning: Watch more debate highlights: Huntsman Mocks Cain: ‘I Thought 9-9-9 Was Price Of Pizza’ Bachmann: ‘Turn 9-9-9 Upside Down’ Herman Cain To Moderator: ‘The Problem With Your Anlaysis Is That It Is Incorrect’ Romney Jabs Charlie Rose…

Assessment: Herman Cain's 9-9-9 plan appears as article subject versus only in links to debate clips under a Romney headline; no common occurrence is established.

### triplet_001678 — same_specific_story

Shared source ID(s): 813094.

- Anchor **left_004657 (left)** — Iraq Leader Says No Pardons for Attacks on Soldiers
  - Beginning: BAGHDAD, Iraq, June 27 —Prime Minister Nuri Kamal al-Maliki of Iraq said today that attacks on American soldiers would not be pardoned under the rules of a new Iraqi amnesty plan. In his first meeting with Western reporters since …
- Positive **right_005076 (right)** — Get-Out-of-Jail Card for Terrorists?
  - Beginning: U.S. lawmakers called a proposal to give amnesty to Iraq's insurgents a "get-out-of-jail-free" policy, and demanded that President Bush reject the plan. Iraqi Prime Minister Nouri al-Maliki's proposed reconciliation plan would gra…

Assessment: The proposed Iraqi insurgent amnesty plan and the same controversy over whether attackers of U.S. soldiers would be pardoned.

### triplet_001705 — same_specific_story

Shared source ID(s): 560860, 560867.

- Anchor **right_004887 (right)** — Danny Glover Teams with Moveon.org to Nominate Paul Krugman for Treasury Post
  - Beginning: New York Times columnist Paul Krugman once suggested an alien invasion would be a grand way of turning the country's economic fortunes around. Now, actor Danny Glover thinks the far-left columnist is the right candidate to replace…
- Positive **left_004473 (left)** — White House Krugman Petition
  - Beginning: I received an email from Danny Glover on behalf of MoveOn.org, asking me to sign a petition through SignOn supporting the nomination of Pulitzer Prize winning economist Paul Krugman for Secretary of the Treasury. I wondered why I …

Assessment: Danny Glover/MoveOn push to nominate Paul Krugman for Treasury secretary; the left article describes a response petition.

### triplet_001943 — same_specific_story

Shared source ID(s): 909662.

- Anchor **left_005634 (left)** — The James Comey memo is an existential threat to Donald Trump's presidency
  - Beginning: (CNN)The reporting coming out of The New York Times and CNN is explosive: Deposed FBI Director James Comey wrote a memo following a February 14 meeting with Donald Trump in which he says the President told him "I hope you can let …
- Positive **right_006081 (right)** — Krauthammer: “Who is going to step out now and defend the president in these denials?” | Special Report | Bret Baier
  - Beginning: Krauthammer: “Who is going to step out now and defend the president in these denials?” Charles Krauthammer told viewers Tuesday on “Special Report with Bret Baier” that when it comes to breaking allegations about President Trump b…

Assessment: The Comey memo allegation that Trump asked him to end the Michael Flynn investigation.

## Implication

The earlier count of “10,000 events containing both ideologies” describes 10,000 **source pairing IDs**. It overstates the number of independently verified real-world events. The saved triplet evaluation used IDs as positives; its numbers are mathematically reproducible, while the semantic positive labels contain observable noise. No triplet scores were changed by this audit.
