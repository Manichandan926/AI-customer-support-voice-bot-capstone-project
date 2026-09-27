# Evaluation: 105 held-out questions, rule engine only (offline, no AI)

| metric | Overall | English | Telugu | Hindi |
|---|---|---|---|---|
| in-scope questions | 93 | 61 | 16 | 16 |
| accuracy (answered correctly) | 68.8% | 62.3% | 75.0% | 87.5% |
| resolved incl. 'did you mean?' | 88.2% | 82.0% | 100.0% | 100.0% |
| answer precision (right when it answers) | 88.9% | 82.6% | 100.0% | 100.0% |
| wrong answers | 8 | 8 | 0 | 0 |
| missed (sent to human) | 3 | 3 | 0 | 0 |
| off-topic correctly rejected | 91.7% | 87.5% | 100.0% | 100.0% |
| language detected correctly | 100.0% | 100.0% | 100.0% | 100.0% |
| avg latency (ms) | 0.85 | 0.58 | 1.58 | 1.20 |
| p95 latency (ms) | 1.15 | 0.57 | 1.28 | 1.21 |

## Not answered correctly (30)

| lang | question | expected | outcome | bot did |
|---|---|---|---|---|
| en | whats the status of my package | order_status | clarify | clarify order_status |
| en | i want to know where my shipment is | order_status | clarify | clarify order_status |
| en | how can i cancel something i bought | cancel_order | wrong | clarify price_drop |
| en | i entered the wrong shipping adress | change_address | clarify | clarify change_address |
| en | do i have to pay for delivery | shipping_cost | wrong | answer cod |
| en | my package still hasnt arrived | delayed_order | wrong | answer damaged_item |
| en | the delivery is taking too long | delayed_order | wrong | answer shipping_time |
| en | the phone screen came cracked | damaged_item | missed | escalate  |
| en | can i return things after 20 days | return_policy | clarify | clarify return_policy |
| en | steps to return a product | how_to_return | wrong | answer return_policy |
| en | where does the refund money go | refund_method | wrong | answer refund_time |
| en | can i pay cash when it arrives | cod | clarify | clarify cod |
| en | my card was charged but the order failed | payment_failed | missed | escalate  |
| en | the price went down right after i bought it | price_drop | missed | escalate  |
| en | how do i register on your site | create_account | wrong | clarify seller |
| en | i cant remember my password | reset_password | clarify | clarify reset_password |
| en | unable to login to my account | reset_password | clarify | clarify reset_password |
| en | someone placed orders from my account without me | account_hacked | clarify | clarify account_hacked |
| en | is there a warranty on these headphones | warranty | clarify | clarify warranty |
| en | when will this item be back in stock | out_of_stock | clarify | clarify out_of_stock |
| en | which size should i order | size_guide | wrong | answer modify_order |
| en | what time does customer support open | working_hours | clarify | clarify working_hours |
| en | do you share my personal information | privacy | clarify | clarify privacy |
| en | what is the bitcoin price today | escalate | false_answer | clarify shipping_cost |
| te | ఆర్డర్ ను రద్దు చేయాలనుకుంటున్నాను | cancel_order | clarify | clarify cancel_order |
| te | డెలివరీకి ఎన్ని రోజులు పడుతుంది | shipping_time | clarify | clarify shipping_time |
| te | డెలివరీ ఛార్జీలు ఉంటాయా | shipping_cost | clarify | clarify shipping_cost |
| te | నాకు వచ్చిన వస్తువు పగిలిపోయింది | damaged_item | clarify | clarify damaged_item |
| hi | आपने गलत सामान भेज दिया | wrong_item | clarify | clarify wrong_item |
| hi | mera paisa wapas kab aayega | refund_time | clarify | clarify refund_time |
