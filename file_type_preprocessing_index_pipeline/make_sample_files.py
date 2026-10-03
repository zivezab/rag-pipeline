"""Generates small sample .txt, .md, and .pdf recipe files for the tutorial.

The original tutorial downloads a folder of mixed file types from Google
Drive via `gdown`. That dependency is fragile in automated/offline
environments (Google Drive folder links can require auth, rate-limit, or
simply become unavailable), so this script creates equivalent local sample
files instead. If you'd rather use the original Google Drive folder, you
can still do so manually and point `preprocessing_pipeline.py` at that
directory via the RECIPE_FILES_DIR environment variable.
"""

from pathlib import Path

from fpdf import FPDF

OUTPUT_DIR = Path("recipe_files")

TXT_RECIPE = """Vegan Keto Eggplant Lasagna

Ingredients:
- 2 large eggplants
- Salt, to taste
- 1/2 cup store-bought vegan mozzarella for topping
- Pesto: 4 oz basil, 1/4 cup almonds, 1/4 cup nutritional yeast, 1/4 cup olive oil
- Spinach tofu ricotta: 14 oz firm tofu, 10 oz spinach, juice of 1 lemon, garlic powder, salt
- Macadamia nut cheese: 1 cup macadamia nuts, 1/4 cup nutritional yeast, 1/4 cup olive oil,
  1 tsp garlic powder, juice of half a lemon, salt to taste

Instructions:
1. Slice the eggplants lengthwise and salt them to draw out moisture.
2. Layer the eggplant slices with pesto, spinach tofu ricotta, and macadamia nut cheese
   in a 9x13 casserole dish.
3. Top with vegan mozzarella and bake until golden and bubbling.
"""

MD_RECIPE = """# Vegan Persimmon Flan

## Ingredients

- 2 average-sized fuyu persimmons, strained
- 1 tbsp cornstarch
- 1/2 tsp agar agar
- 1 tbsp agave nectar, or to taste
- 2 tbsp granulated sugar
- 1/4 cup coconut creme
- 1/2 cup almond milk
- 1/2 tsp vanilla

## Instructions

1. Blend the strained persimmon pulp with coconut creme, almond milk, and vanilla.
2. Whisk in the cornstarch, agar agar, agave nectar, and sugar.
3. Pour into ramekins and set using a hot water bath method.
4. Chill before serving.
"""

PDF_RECIPE_TITLE = "Vegan Hemp Cheese"
PDF_RECIPE_BODY = """Ingredients:
- 1/2 cup sunflower seeds
- 1/2 cup hemp hearts
- 1.5 teaspoons miso paste
- 1 tsp nutritional yeast
- 1/4 cup rejuvelac
- 1/4 teaspoon salt, or to taste

Instructions:
1. Blend all ingredients in a food processor until smooth.
2. Transfer to a clean glass bowl, cover with a dish towel secured by a rubber band.
3. Let ferment at room temperature for 12-24 hours, then refrigerate.
"""


def write_txt_recipe(output_dir: Path) -> Path:
    path = output_dir / "vegan_keto_eggplant_lasagna.txt"
    path.write_text(TXT_RECIPE, encoding="utf-8")
    return path


def write_md_recipe(output_dir: Path) -> Path:
    path = output_dir / "vegan_persimmon_flan.md"
    path.write_text(MD_RECIPE, encoding="utf-8")
    return path


def write_pdf_recipe(output_dir: Path) -> Path:
    path = output_dir / "vegan_hemp_cheese.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=16)
    pdf.cell(0, 10, PDF_RECIPE_TITLE, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, PDF_RECIPE_BODY)
    pdf.output(str(path))
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    txt_path = write_txt_recipe(OUTPUT_DIR)
    md_path = write_md_recipe(OUTPUT_DIR)
    pdf_path = write_pdf_recipe(OUTPUT_DIR)
    for path in (txt_path, md_path, pdf_path):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
