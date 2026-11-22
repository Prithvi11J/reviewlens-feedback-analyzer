"""Generate a realistic week of reviews for PJ's, a (fictional) Chettinad restaurant
with three outlets in Coimbatore: RS Puram, Gandhipuram and Peelamedu.

Writes two files:
  data/sample/reviews_week47.csv          the input: the brief's 5 columns + an optional `outlet`
  data/sample/reviews_week47_labels.csv   the "answer key" (what each review is really about)

The answer key lets us MEASURE accuracy instead of eyeballing it. It is kept in a
separate file so the LLM never sees it.

The data is generated from hand-written review templates with a fixed random seed,
so running this script again always produces the identical files.

PLANTED PATTERNS (a good weekly report should discover all four):
  1. Delivery complaints come mostly from Swiggy (~75%).
  2. Billing complaints (service charge, extra charges) are mostly Tanglish (~70%)
     and mostly on Google (~70%).
  3. A food-poisoning spike: 6 reports about the vanjaram fish fry, Fri 20 and
     Sat 21 Nov, all from the Peelamedu outlet.
  4. Packaging complaints come mostly from the Peelamedu outlet (~70%).
Everything else (platforms, outlets, dates) is random, so any other "pattern" is noise.

Run from the project root:
    python scripts/make_test_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 47
N_REVIEWS = 240
WEEK_START = date(2026, 11, 16)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"

OUTLETS = {"RS Puram": 0.36, "Gandhipuram": 0.36, "Peelamedu": 0.28}
PLATFORMS = ["Google", "Zomato", "Swiggy"]

# (category, sentiment, language) -> review texts written for that combination.
TEMPLATES: dict[tuple[str, str, str], list[str]] = {
    # ---------------- food_quality ----------------
    ("food_quality", "positive", "English"): [
        "The Chettinad chicken biryani at PJ's is the real deal. Spicy, aromatic and the meat was so tender.",
        "Nattu kozhi curry tasted exactly like my grandmother's. Proper pepper kick.",
        "Kothu parotta was perfect, and the salna they give with it is addictive.",
        "Mutton biryani was rich and flavourful, generous portion for the price.",
        "Lunch meals were excellent: rasam, poriyal and the kozhambu all tasted homemade.",
        "Mushroom pepper fry is the best veg dish here. Even my non-veg friends ordered seconds.",
        "Paniyaram with kara chutney was crispy outside and soft inside. Lovely evening snack.",
    ],
    ("food_quality", "negative", "English"): [
        "Biryani rice was undercooked and hard today. Not the PJ's standard we know.",
        "The mutton in the biryani smelled stale. Threw the whole thing away.",
        "Way too much salt in the nattu kozhi curry, could not eat it.",
        "Portion of the mutton biryani has become really small for the same price.",
        "Found a hair in my chicken chettinad gravy. Disgusting.",
        "Kothu parotta was soggy and oily, nothing like before.",
    ],
    ("food_quality", "neutral", "English"): [
        "Food was okay, nothing special. The biryani is average for the hype.",
        "Meals were fine but I have had better Chettinad food in Karaikudi.",
    ],
    ("food_quality", "positive", "Tanglish"): [
        "PJ's biryani semma taste, kandippa marubadiyum order pannuven!",
        "Saapadu romba nalla irundhuchu, veetu saapadu madhiri.",
        "Nattu kozhi kuzhambu vera level, kaaram correct ah irundhuchu.",
        "Kothu parotta and salna combo super, full ah saaptom.",
    ],
    ("food_quality", "negative", "Tanglish"): [
        "Biryani la kozhi vegave illa, romba mosam.",
        "Saapadu la uppu romba jaasthi, saapda mudiyala.",
        "Kuzhambu kettu pochu, smell adichuchu. Waste of money.",
    ],
    ("food_quality", "positive", "Tamil"): [
        "பிரியாணி ரொம்ப சுவையா இருந்துச்சு. மறுபடியும் ஆர்டர் பண்ணுவேன்.",
        "சாப்பாடு வீட்டு சாப்பாடு மாதிரி நல்லா இருந்துச்சு.",
        "நாட்டுக்கோழி குழம்பு அருமை. காரம் சரியா இருந்துச்சு.",
    ],
    ("food_quality", "negative", "Tamil"): [
        "சாப்பாட்டுல முடி இருந்துச்சு. ரொம்ப அருவருப்பா இருந்துச்சு.",
        "குழம்பு கெட்டுப் போயிருந்துச்சு, சாப்பிட முடியல.",
    ],
    # ---------------- delivery ----------------
    ("delivery", "positive", "English"): [
        "Delivered in 25 minutes and the biryani was still steaming hot. Impressed.",
        "Super fast delivery even in the rain. Food reached hot.",
    ],
    ("delivery", "negative", "English"): [
        "Ordered at 8pm, biryani came at 9:15. Fully cold. Very bad.",
        "Delivery took more than an hour. The tracking kept showing 10 minutes.",
        "Rider went to the wrong address and the food came cold.",
        "Late again. Third time this month the dinner order is 45 minutes late.",
        "Order was marked delivered but I never received it. I want a refund.",
        "Waited 80 minutes for dinner, kids slept hungry.",
        "Wrong order delivered. I am vegetarian and got chicken biryani instead of veg meals.",
    ],
    ("delivery", "neutral", "English"): [
        "Delivery was on time, nothing to complain about.",
    ],
    ("delivery", "positive", "Tanglish"): [
        "Delivery semma fast, 20 nimishathula vandhuchu, biryani hot ah irundhuchu.",
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
    # ---------------- price (billing) ----------------
    ("price", "positive", "English"): [
        "Great value for money, the family biryani bucket fed four of us.",
        "Lunch meals at this price is a steal compared to other places in Coimbatore.",
    ],
    ("price", "negative", "English"): [
        "They added a 10% service charge to the bill without asking us.",
        "Prices went up by 40 rupees in one month. Not worth it anymore.",
        "Bill was 120 rupees more than the menu price. Hidden charges.",
        "Charged me twice for the same order, still waiting for the refund.",
    ],
    ("price", "negative", "Tanglish"): [
        "Saapadu nalla irundhuchu but bill la extra charge pottanga.",
        "Service charge kekkama bill la serthutaanga, ketta kooda remove panna maatenga.",
        "Menu la oru vela, bill la oru vela. Hidden charges romba.",
        "Price romba jaasthi, portion romba kammi.",
    ],
    ("price", "negative", "Tamil"): [
        "பில்லில் சர்வீஸ் சார்ஜ் சேர்த்திருக்காங்க. எங்ககிட்ட கேக்கவே இல்லை.",
    ],
    # ---------------- staff ----------------
    ("staff", "positive", "English"): [
        "The waiter at the RS Puram branch was very polite and helped us choose dishes.",
        "Staff were friendly and quick even when the restaurant was full on Sunday.",
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
        "Packing was neat and leak-proof, loved that the meals came in banana leaf wrap.",
        "Very good packaging, the gravy did not spill at all.",
    ],
    ("packaging", "negative", "English"): [
        "Salna leaked all over the bag, the packing was terrible.",
        "Container lid was broken and the biryani spilled everywhere.",
        "Rasam packet burst inside the bag. Poor packaging.",
    ],
    ("packaging", "negative", "Tanglish"): [
        "Parcel full ah leak aagi bag ellam kuzhambu.",
        "Packing romba mosam, box odanju pochu.",
    ],
    ("packaging", "negative", "Tamil"): [
        "பார்சல் கசிஞ்சு பை முழுக்க குழம்பு ஆயிடுச்சு.",
    ],
    # ---------------- app (ordering and payment) ----------------
    ("app", "positive", "English"): [
        "Ordering through PJ's WhatsApp menu is smooth and the payment link works instantly.",
    ],
    ("app", "negative", "English"): [
        "UPI payment failed at checkout but money was debited from my account. Support is useless.",
        "Coupon code is not working on the PJ's order page, tried five times.",
        "The online order page crashes every time I add the family pack to the cart.",
        "Order page keeps logging me out right at the payment step.",
    ],
    ("app", "negative", "Tanglish"): [
        "Online la payment fail aachu but amount cut aagiduchu.",
        "Coupon apply aagave illa, order page la problem.",
    ],
    ("app", "negative", "Tamil"): [
        "ஆன்லைன்ல பேமெண்ட் பெயில் ஆச்சு ஆனா பணம் கட் ஆயிடுச்சு.",
    ],
}

# Answer key for urgency. Following the strict rule, "high" means one of:
#   health/safety (spoiled food, hair/foreign object, food poisoning),
#   money taken wrongly (refund demand, double charge, debited but failed),
#   abusive/rude/hostile staff, wrong or missing order.
# Every other negative review is "medium"; positive/neutral reviews are "low".
HIGH_URGENCY_TEXTS = {
    "The mutton in the biryani smelled stale. Threw the whole thing away.",
    "Found a hair in my chicken chettinad gravy. Disgusting.",
    "Kuzhambu kettu pochu, smell adichuchu. Waste of money.",
    "சாப்பாட்டுல முடி இருந்துச்சு. ரொம்ப அருவருப்பா இருந்துச்சு.",
    "குழம்பு கெட்டுப் போயிருந்துச்சு, சாப்பிட முடியல.",
    "Order was marked delivered but I never received it. I want a refund.",
    "Wrong order delivered. I am vegetarian and got chicken biryani instead of veg meals.",
    "Charged me twice for the same order, still waiting for the refund.",
    "UPI payment failed at checkout but money was debited from my account. Support is useless.",
    "Online la payment fail aachu but amount cut aagiduchu.",
    "ஆன்லைன்ல பேமெண்ட் பெயில் ஆச்சு ஆனா பணம் கட் ஆயிடுச்சு.",
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
    "food_quality": 0.38, "delivery": 0.22, "price": 0.12,
    "staff": 0.10, "packaging": 0.10, "app": 0.08,
}
LANGUAGE_WEIGHTS = {"English": 0.60, "Tanglish": 0.28, "Tamil": 0.12}
SENTIMENT_WEIGHTS = {"positive": 0.55, "negative": 0.35, "neutral": 0.10}

OPENERS = {
    "English": ["", "", "", "Ordered for lunch today. ", "Second time ordering from PJ's. ", "Visited with family. "],
    "Tanglish": ["", "", "Inniki lunch order panninen. ", "Family oda PJ's ku ponom. "],
    "Tamil": ["", "", "இன்னைக்கு மதியம் ஆர்டர் பண்ணேன். "],
}

# Pattern 3: the food-poisoning spike (vanjaram fish fry, Peelamedu, Fri/Sat only).
SPIKE_OUTLET = "Peelamedu"
SPIKE_DATES = [date(2026, 11, 20), date(2026, 11, 21)]
SPIKE_REVIEWS = [
    ("English", "Had the vanjaram fish fry on Friday night, whole family had vomiting and stomach pain."),
    ("English", "Food poisoning after eating your fish fry yesterday. Had to go to the hospital."),
    ("English", "My son got sick after the fish fry from PJ's Peelamedu. Please check the fish quality."),
    ("Tanglish", "Fish fry saapta udane vayiru vali arambichiruchu, full night vomiting."),
    ("Tanglish", "Vanjaram fry saaptu food poisoning aagiduchu, hospital ku ponom."),
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
                   language: str, text: str) -> tuple[str, str, str, str]:
    """Bias platform/outlet/language for planted patterns 1, 2 and 4.

    Returns (platform, outlet, language, text).
    """
    platform = rng.choice(PLATFORMS)
    outlet = pick(rng, OUTLETS)
    if (category, sentiment) == ("delivery", "negative") and rng.random() < 0.75:
        platform = "Swiggy"  # pattern 1
    if (category, sentiment) == ("price", "negative"):  # pattern 2
        if rng.random() < 0.70:
            language = "Tanglish"
            text = rng.choice(TEMPLATES[("price", "negative", "Tanglish")])
        if rng.random() < 0.70:
            platform = "Google"
    if (category, sentiment) == ("packaging", "negative") and rng.random() < 0.70:
        outlet = "Peelamedu"  # pattern 4
    return platform, outlet, language, text


def main() -> None:
    rng = random.Random(SEED)
    rows, labels = [], []

    def add(category, sentiment, language, text, platform, outlet, day, urgency=None):
        review_id = f"R-{2001 + len(labels)}"
        rows.append({
            "review_id": review_id,
            "date": day.isoformat(),
            "platform": platform,
            "rating": rng.choice(RATING_BY_SENTIMENT[sentiment]),
            "review_text": rng.choice(OPENERS[language]) + text,
            "outlet": outlet,
        })
        labels.append({"review_id": review_id, "category": category, "sentiment": sentiment,
                       "language": language, "urgency": urgency or expected_urgency(sentiment, text)})

    for _ in range(N_REVIEWS - len(SPIKE_REVIEWS)):
        category, sentiment, language, text = choose_template(rng)
        platform, outlet, language, text = apply_patterns(rng, category, sentiment, language, text)
        add(category, sentiment, language, text, platform, outlet,
            WEEK_START + timedelta(days=rng.randrange(7)))

    for language, text in SPIKE_REVIEWS:  # pattern 3
        add("food_quality", "negative", language, text, rng.choice(PLATFORMS), SPIKE_OUTLET,
            rng.choice(SPIKE_DATES), urgency="high")

    # Messy rows that the validation step must handle:
    rows.append({"review_id": "R-9001", "date": "2026-11-18", "platform": "Google", "rating": 5,
                 "review_text": "", "outlet": "RS Puram"})
    rows.append({"review_id": "R-9002", "date": "2026-11-19", "platform": "Zomato", "rating": 4,
                 "review_text": "   ", "outlet": "Gandhipuram"})
    rows.append(dict(rows[10]))  # an exact duplicate of an earlier row (same review_id)

    rng.shuffle(rows)  # spike rows should not all sit at the end of the file
    rows.sort(key=lambda r: r["date"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # utf-8-sig adds a BOM so Excel displays Tamil text correctly.
    with open(OUT_DIR / "reviews_week47.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "date", "platform", "rating", "review_text", "outlet"])
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_DIR / "reviews_week47_labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "category", "sentiment", "language", "urgency"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Wrote {len(rows)} rows ({N_REVIEWS} real + 3 messy) to {OUT_DIR}")


if __name__ == "__main__":
    main()
