from jinja2 import Template


def get_default_prompt_template():
    return """
{% if age < 0 %}
    {% set age_group = '' %}
{% elif age >=0 and age < 14 %}
    {% set age_group = 'child' %}
{% elif age >= 14 and age <= 50 %}
    {% set age_group = 'young' %}
{% elif age > 50 and age <= 65 %}
    {% set age_group = 'middle age' %}
{% elif age > 65 %}
    {% set age_group = 'old' %}
{% else %}
    {% set age_group = 'old' %}
{% endif %}
{{ age_group }} {{ race }} {{ gender }}
""".strip()


def generate_prompt(prompt_template, analysis):
    rendered_strings = []
    for context in analysis:
        print("processing context: ", context)
        template = Template(prompt_template)
        rendered_string = template.render(context).strip()
        rendered_strings.append(rendered_string)
    print(rendered_strings)

    return " BREAK\n".join(rendered_strings)
