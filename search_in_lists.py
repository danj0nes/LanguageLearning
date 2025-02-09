import os

QUIT_TOKEN = ":q"


def search_in_file(file_path, search_term):
    """Searches for the term in a file and returns all matching French-English pairs."""
    results = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            french_phrase = lines[i].strip()
            english_definition = lines[i + 1].strip()

            if (
                search_term.lower() in french_phrase.lower()
                or search_term.lower() in english_definition.lower()
            ):
                results.append((french_phrase, english_definition))
    return results


def search_in_all_files(search_term):
    """Searches for the term in all .txt files within the script's folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    all_results = {}

    for root, _, files in os.walk(script_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                results = search_in_file(file_path, search_term)
                if results:
                    all_results[filename] = results

    return all_results


def display_results(search_term, results):
    """Displays the results in the specified format."""
    if results:
        print(f"\nHere are all the occurrences of '{search_term}':\n")
        for filename, occurences in results.items():
            print(f"\t{filename}")
            for french_phrase, english_definition in occurences:
                print(f"{french_phrase}\n{english_definition}\n")
    else:
        print(f"\nNo occurrences of '{search_term}' found.\n")


def main():
    print(f"Type '{QUIT_TOKEN}' to quit\n")
    while True:
        search_term = input("\nEnter the word or phrase to search for: ")
        if search_term == QUIT_TOKEN:
            break
        results = search_in_all_files(search_term)
        display_results(search_term, results)


if __name__ == "__main__":
    main()
