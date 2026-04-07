from audioencoder import AudioProcessor 
from audioencoder import Finder

def main():
    print("Hello from project!")
    processor = AudioProcessor()
    finder = Finder()
    file_path = finder.fuzzy_find_file("./")
    if file_path is None: raise ValueError("Path is emty")
    files = finder.get_all_files(file_path)
    print(f"Processing => {files}")
    processor.visualize_space(files)

if __name__ == "__main__":
    main()
