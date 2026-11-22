"""Generate a realistic week of reviews to test ReviewLens with our own data.

Writes two files:
  data/sample/reviews_week47.csv          the input, in the brief's exact format
  data/sample/reviews_week47_labels.csv   the "answer key" (what each review is really about)

The answer key lets us MEASURE accuracy instead of eyeballing it. It is kept in a
separate file so the LLM never sees it.

The data is generated from hand-written review templates with a fixed random seed,
so running this script again always produces the identical files.

PLANTED PATTERNS (a good weekly report should discover all three):
  1. Delivery complaints come mostly from Swiggy (~75%).
  2. Extra-charge / price complaints are mostly Tanglish (~70%) and mostly on Google (~70%).
  3. A food-poisoning spike: 6 reports about the fish fry on Fri 20 and Sat 21 Nov.
Everything else (platforms, dates) is random, so any other "pattern" is noise.

Run from the project root:
    python scripts/make_test_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_REVIEWS = 240
WEEK_START = date(2026, 11, 16)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"

# (category, sentiment, language) -> review texts written for that combination.
TEMPLATES: dict[tuple[str, str, str], list[str]] = {
    # ---------------- food_quality ----------------
    ("food_quality", "positive", "English"): [
        "Biryani was amazing, perfectly spiced and the meat was so tender.",
        "Best chicken 65 I have had in Coimbatore. Crispy and fresh.",
        "Parotta and salna combo was superb, tasted homemade.",
        "Mutton biryani was rich and flavourful, portion was generous too.",
        "Filter coffee and ghee roast were top class this morning.",
    ],
    ("food_quality", "negative", "English"): [
        "Biryani rice was undercooked and hard. Very disappointed.",
        "Chicken smelled stale. Threw the whole thing away.",
        "Way too much salt in the gravy, could not eat it.",
        "Portion size has become really small for the same price.",
        "Found a hair in my curry. Disgusting.",
    ],
    ("food_quality", "neutral", "English"): [
        "Food was okay, nothing special. Average taste.",
        "Biryani was fine but I have had better at other places.",
    ],
    ("food_quality", "positive", "Tanglish"): [
        "Biryani semma taste, kandippa marubadiyum order pannuven!",
        "Saapadu romba nalla irundhuchu, veetu saapadu madhiri.",
        "Chicken 65 vera level, crispy ah irundhuchu.",
    ],
    ("food_quality", "negative", "Tanglish"): [
        "Biryani la kozhi vegave illa, romba mosam.",
        "Saapadu la uppu romba jaasthi, saapda mudiyala.",
        "Curry kettu pochu, smell adichuchu. Waste of money.",
    ],
    ("food_quality", "positive", "Tamil"): [
        "பிரியாணி ரொம்ப சுவையா இருந்துச்சு. மறுபடியும் ஆர்டர் பண்ணுவேன்.",
        "சாப்பாடு வீட்டு சாப்பாடு மாதிரி நல்லா இருந்துச்சு.",
    ],
    ("food_quality", "negative", "Tamil"): [
        "சாப்பாட்டுல முடி இருந்துச்சு. ரொம்ப அருவருப்பா இருந்துச்சு.",
        "கறி கெட்டுப் போயிருந்துச்சு, சாப்பிட முடியல.",
    ],
    # ---------------- delivery ----------------
    ("delivery", "positive", "English"): [
        "Delivered in 25 minutes and the food was still hot. Impressed.",
        "Super fast delivery even in the rain. Thanks to the rider.",
    ],
    ("delivery", "negative", "English"): [
        "Ordered at 8pm, came at 9:15. Food fully cold. Very bad.",
        "Delivery took more than an hour. The app kept showing 10 minutes.",
        "Rider went to the wrong address and food came cold.",
        "Late again. Third time this week the order is 45 minutes late.",
        "Order was marked delivered but I never received it. I want a refund.",
        "Waited 80 minutes for dinner, kids slept hungry.",
        "Wrong order delivered. I am vegetarian and got chicken biryani instead of veg meals.",
    ],
    ("delivery", "neutral", "English"): [
        "Delivery was on time, nothing to complain about.",
    ],
    ("delivery", "positive", "Tanglish"): [
        "Delivery semma fast, 20 nimishathula vandhuchu.",
    ],
    ("delivery", "negative", "Tanglish"): [
        "Order panni 1 hour aachu, innum varala. Romba late.",
        "Saapadu vandhapo full ah aaripochu, delivery romba late.",
        "Delivery boy thappana address ku poitaru, late ah vandhuchu.",
    ],
    ("delivery", "negative", "Tamil"): [
        "ஒரு மணி நேரம் கழிச்சு தான் டெலிவரி வந்துச்சு. சாப்பாடு ஆறிப் போச்சு.",
        "டெலிவரி ரொம்ப லேட், இனிமே ஆர்டர் பண்ண மாட்டேன்.",
    ],
    # ---------------- price ----------------
    ("price", "positive", "English"): [
        "Great value for money, family meal fed four of us.",
        "Combo offer is worth it, cheaper than other places nearby.",
    ],
    ("price", "negative", "English"): [
        "They added a packing charge and service charge that were not shown on the menu.",
        "Prices increased by 40 rupees in one month. Not worth it anymore.",
        "Bill was 120 rupees more than the menu price. Hidden charges.",
        "Charged me twice for the same order, still waiting for the refund.",
    ],
    ("price", "negative", "Tanglish"): [
        "Saapadu nalla irundhuchu but bill la extra charge pottanga.",
        "Menu la oru vela, bill la oru vela. Hidden charges romba.",
        "Price romba jaasthi, portion romba kammi.",
    ],
    ("price", "negative", "Tamil"): [
        "பில்லில் கூடுதல் கட்டணம் போட்டிருக்காங்க. மெனுவில் இது இல்லை.",
    ],
    # ---------------- staff ----------------
    ("staff", "positive", "English"): [
        "The waiter Ravi was very polite and helped us choose dishes.",
        "Staff were friendly and quick even when the restaurant was full.",
    ],
    ("staff", "negative", "English"): [
        "The cashier was very rude and shouted at my mother.",
        "Manager argued with us instead of listening. Never coming back.",
        "Staff ignored us for 20 minutes, no one took our order.",
    ],
    ("staff", "negative", "Tanglish"): [
        "Waiter romba rude ah pesinaru, mariyadhaiye illa.",
        "Staff kandukave illa, 30 nimisham wait panninom.",
    ],
    ("staff", "negative", "Tamil"): [
        "ஊழியர்கள் மரியாதை இல்லாம பேசினாங்க. ரொம்ப மோசம்.",
    ],
    ("staff", "positive", "Tamil"): [
        "ஊழியர்கள் ரொம்ப அன்பா கவனிச்சாங்க. நன்றி.",
    ],
    # ---------------- packaging ----------------
    ("packaging", "positive", "English"): [
        "Packing was neat and leak-proof, loved the eco-friendly boxes.",
        "Very good packaging, gravy did not spill at all.",
    ],
    ("packaging", "negative", "English"): [
        "Gravy leaked all over the bag, the packing was terrible.",
        "Container lid was broken and rice spilled everywhere.",
        "Sambar packet burst inside the bag. Poor packaging.",
    ],
    ("packaging", "negative", "Tanglish"): [
        "Parcel full ah leak aagi bag ellam kozhambu.",
        "Packing romba mosam, box odanju pochu.",
    ],
    ("packaging", "negative", "Tamil"): [
        "பார்சல் கசிஞ்சு பை முழுக்க குழம்பு ஆயிடுச்சு.",
    ],
    # ---------------- app ----------------
    ("app", "positive", "English"): [
        "Ordering through your app is smooth and the live tracking is accurate.",
    ],
    ("app", "negative", "English"): [
        "Payment failed but money was debited from my account. App support is useless.",
        "Coupon code is not working in the app, tried five times.",
        "App crashes every time I try to add items to the cart.",
        "App keeps logging me out right at the payment step.",
    ],
    ("app", "negative", "Tanglish"): [
        "App la payment fail aachu but amount cut aagiduchu.",
        "Coupon apply aagave illa, app la problem.",
    ],
    ("app", "negative", "Tamil"): [
        "ஆப்ல பேமெண்ட் பெயில் ஆச்சு ஆனா பணம் கட் ஆயிடுச்சு.",
    ],
}

# Answer key for urgency. Following the strict rule, "high" means one of:
#   health/safety (spoiled food, hair/foreign object, food poisoning),
#   money taken wrongly (refund demand, double charge, debited but failed),
#   abusive/rude/hostile staff, wrong or missing order.
# Every other negative review is "medium"; positive/neutral reviews are "low".
HIGH_URGENCY_TEXTS = {
    "Chicken smelled stale. Threw the whole thing away.",
    "Found a hair in my curry. Disgusting.",
    "Curry kettu pochu, smell adichuchu. Waste of money.",
    "சாப்பாட்டுல முடி இருந்துச்சு. ரொம்ப அருவருப்பா இருந்துச்சு.",
    "கறி கெட்டுப் போயிருந்துச்சு, சாப்பிட முடியல.",
    "Order was marked delivered but I never received it. I want a refund.",
    "Wrong order delivered. I am vegetarian and got chicken biryani instead of veg meals.",
    "Charged me twice for the same order, still waiting for the refund.",
    "Payment failed but money was debited from my account. App support is useless.",
    "App la payment fail aachu but amount cut aagiduchu.",
    "ஆப்ல பேமெண்ட் பெயில் ஆச்சு ஆனா பணம் கட் ஆயிடுச்சு.",
    "The cashier was very rude and shouted at my mother.",
    "Manager argued with us instead of listening. Never coming back.",
    "Waiter romba rude ah pesinaru, mariyadhaiye illa.",
    "ஊழியர்கள் மரியாதை இல்லாம பேசினாங்க. ரொம்ப மோசம்.",
}


def expected_urgency(sentiment: str, text: str) -> str:
    if text in HIGH_URGENCY_TEXTS:
        return "high"
    return "medium" if sentiment == "negative" else "low"


# How often each category appears, roughly like a real restaurant's week.
CATEGORY_WEIGHTS = {
    "food_quality": 0.36, "delivery": 0.22, "price": 0.12,
    "staff": 0.10, "packaging": 0.10, "app": 0.10,
}
LANGUAGE_WEIGHTS = {"English": 0.60, "Tanglish": 0.28, "Tamil": 0.12}
SENTIMENT_WEIGHTS = {"positive": 0.55, "negative": 0.35, "neutral": 0.10}

OPENERS = {
    "English": ["", "", "", "Ordered for lunch today. ", "Second time ordering. ", "Visited with family. "],
    "Tanglish": ["", "", "Inniki lunch order panninen. ", "Family oda ponom. "],
    "Tamil": ["", "", "இன்னைக்கு மதியம் ஆர்டர் பண்ணேன். "],
}
PLATFORMS = ["Google", "Zomato", "Swiggy", "Amazon"]

# Pattern 3: the food-poisoning spike (all about the fish fry, Fri/Sat only).
SPIKE_DATES = [date(2026, 11, 20), date(2026, 11, 21)]
SPIKE_REVIEWS = [
    ("English", "Had the fish fry on Friday night, whole family had vomiting and stomach pain."),
    ("English", "Food poisoning after eating your fish fry yesterday. Had to go to the hospital."),
    ("English", "My son got sick after the fish fry from your restaurant. Please check the fish quality."),
    ("Tanglish", "Fish fry saapta udane vayiru vali arambichiruchu, full night vomiting."),
    ("Tanglish", "Fish fry saaptu food poisoning aagiduchu, hospital ku ponom."),
    ("Tamil", "மீன் வறுவல் சாப்பிட்ட பிறகு வயிற்று வலி, வாந்தி. ரொம்ப கஷ்டமா போச்சு."),
]
RATING_BY_SENTIMENT = {"positive": [4, 5, 5], "neutral": [3, 3, 4], "negative": [1, 1, 2, 3]}


def pick(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def choose_template(rng: random.Random) -> tuple[str, str, str, str]:
    """Pick a (category, sentiment, language) combo that has templates, then a text."""
    while True:
        key = (pick(rng, CATEGORY_WEIGHTS), pick(rng, SENTIMENT_WEIGHTS), pick(rng, LANGUAGE_WEIGHTS))
        if key in TEMPLATES:
            return (*key, rng.choice(TEMPLATES[key]))


def apply_patterns(rng: random.Random, category: str, sentiment: str,
                   language: str, text: str) -> tuple[str, str, str]:
    """Bias platform/language for the planted patterns 1 and 2. Returns (platform, language, text)."""
    platform = rng.choice(PLATFORMS)
    if (category, sentiment) == ("delivery", "negative") and rng.random() < 0.75:
        platform = "Swiggy"  # pattern 1
    if (category, sentiment) == ("price", "negative"):  # pattern 2
        if rng.random() < 0.70:
            language = "Tanglish"
            text = rng.choice(TEMPLATES[("price", "negative", "Tanglish")])
        if rng.random() < 0.70:
            platform = "Google"
    return platform, language, text


def main() -> None:
    rng = random.Random(SEED)
    rows, labels = [], []

    def add(category, sentiment, language, text, platform, day, urgency=None):
        review_id = f"R-{2001 + len(labels)}"
        rows.append({
            "review_id": review_id,
            "date": day.isoformat(),
            "platform": platform,
            "rating": rng.choice(RATING_BY_SENTIMENT[sentiment]),
            "review_text": rng.choice(OPENERS[language]) + text,
        })
        labels.append({"review_id": review_id, "category": category, "sentiment": sentiment,
                       "language": language, "urgency": urgency or expected_urgency(sentiment, text)})

    for _ in range(N_REVIEWS - len(SPIKE_REVIEWS)):
        category, sentiment, language, text = choose_template(rng)
        platform, language, text = apply_patterns(rng, category, sentiment, language, text)
        add(category, sentiment, language, text, platform, WEEK_START + timedelta(days=rng.randrange(7)))

    for language, text in SPIKE_REVIEWS:  # pattern 3
        add("food_quality", "negative", language, text, rng.choice(PLATFORMS), rng.choice(SPIKE_DATES),
            urgency="high")

    # Messy rows that the validation step must handle:
    rows.append({"review_id": "R-9001", "date": "2026-11-18", "platform": "Google", "rating": 5, "review_text": ""})
    rows.append({"review_id": "R-9002", "date": "2026-11-19", "platform": "Zomato", "rating": 4, "review_text": "   "})
    rows.append(dict(rows[10]))  # an exact duplicate of an earlier row (same review_id)

    rng.shuffle(rows)  # spike rows should not all sit at the end of the file
    rows.sort(key=lambda r: r["date"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # utf-8-sig adds a BOM so Excel displays Tamil text correctly.
    with open(OUT_DIR / "reviews_week47.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "date", "platform", "rating", "review_text"])
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_DIR / "reviews_week47_labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "category", "sentiment", "language", "urgency"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Wrote {len(rows)} rows ({N_REVIEWS} real + 3 messy) to {OUT_DIR}")


if __name__ == "__main__":
    main()
