import re
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_option_menu import option_menu
from anytree import Node, RenderTree

# Set page config at the top
st.set_page_config(page_title="C Lexer Analyzer", layout="wide", page_icon="🧠")

# ---------------------------
# Session State for Persistence
# ---------------------------
if 'tokens' not in st.session_state:
    st.session_state.tokens = []
    st.session_state.errors = []
    st.session_state.code = ''

# ---------------------------
# Regex Definitions
# ---------------------------
token_specs = [
    ('Library',              r'#include[ \t]*<[^>]+>'),
    ('Line_Comment',         r'//[^\n]*'),
    ('Block_Comment',        r'/\*.*?\*/'),
    ('Access_Specifier',     r'\b(private|protected|public)\b'),
    ('Data_Type',            r'\b(int|float|double|char|bool|string|long|short|void)\b'),
    ('Keyword',              r'\b(if|else|while|for|return|break|continue|switch|case|default|sizeof|do|goto|enum|typedef|struct|class|const|static|volatile|signed|unsigned|try|catch|throw|new|delete)\b'),
    ('Bracket_Parenthesis',  r'[{}\[\]()]'),
    ('Delimiter',            r'[;,:]'),
    ('Assignment_Operator',  r'='),
    ('Increment_Decrement_Operator', r'\+\+|--'),
    ('Arithmetic_Operator',  r'[+\-*/%]'),
    ('Relational_Operator',  r'(==|!=|<=|>=|<|>)'),
    ('Logical_Operator',     r'(\|\||&&|!)'),
    ('Bitwise_Operator',     r'(&|\||\^|~|<<|>>)'),
    ('Float_Constant',       r'\b\d+\.\d+\b'),
    ('Integer_Constant',     r'\b\d+\b'),
    ('Character_Constant',   r"'([^\\']|\\.)'"),
    ('String_Literal',       r'"([^\\"]|\\.)*"'),
    ('Identifier',           r'\b[A-Za-z_][A-Za-z_0-9]*\b'),
    ('Whitespace',           r'[ \t\r]+'),
    ('Newline',              r'\n'),
    ('Unknown_Token',        r'.'),
]

compiled_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specs), re.DOTALL)

# ---------------------------
# Lexer Logic
# ---------------------------


def lex_code(source_code):
    tokens = []
    line_no = 1
    errors = []
    for match in compiled_regex.finditer(source_code):
        kind = match.lastgroup
        value = match.group()

        if kind == 'Whitespace':
            continue
        elif kind == 'Newline':
            line_no += 1
        elif kind == 'Unknown_Token':
            errors.append((line_no, 'Unknown Token', value))
        else:
            tokens.append((line_no, kind, value))
    return tokens, errors

# ---------------------------
# Token Statistics
# ---------------------------


def get_token_stats(tokens):
    stats = {
        'Keyword': 0,
        'Identifier': 0,
        'Constant': 0,
        'Operator': 0,
    }
    for _, kind, _ in tokens:
        if kind == 'Keyword':
            stats['Keyword'] += 1
        elif kind == 'Identifier':
            stats['Identifier'] += 1
        elif 'Constant' in kind:
            stats['Constant'] += 1
        elif 'Operator' in kind:
            stats['Operator'] += 1
    return stats

# ---------------------------
# Parse Tree Generator (Improved)
# ---------------------------


def generate_parse_tree(tokens):
    """
    Build a detailed parse tree for a simple subset of C-like code:
      - Declarations (int/float)
      - Comments
      - Standalone Assignments (x = x + 1;)
      - If statements (with inner assignment)
      - Return statements
      - End marker
    """
    program = Node("Program")
    i = 0
    n = len(tokens)

    while i < n:
        _, kind, val = tokens[i]

        # 1) Skip semicolons, commas, and any parentheses/braces
        if kind in ('Delimiter', 'Bracket_Parenthesis'):
            i += 1
            continue

        # 2) Declarations: int x;  float x = 3.14;
        if kind == 'Data_Type' and val in ('int', 'float'):
            decl = Node(f"Declaration({val})", parent=program)
            i += 1
            _, _, name = tokens[i]  # identifier
            Node(f"ID: {name}", parent=decl)
            i += 1
            if i < n and tokens[i][1] == 'Assignment_Operator':
                i += 1
                _, _, const = tokens[i]
                ctype = 'Float_Constant' if val == 'float' else 'Integer_Constant'
                Node(f"{ctype}: {const}", parent=decl)
                i += 1
            continue

        # 3) Comments
        if kind in ('Line_Comment', 'Block_Comment'):
            Node(f"Comment: {val.strip()}", parent=program)
            i += 1
            continue

        # 4) Standalone Assignment: x = x + 1;
        if kind == 'Identifier' and i + 1 < len(tokens) and tokens[i + 1][1] == 'Assignment_Operator':
            assign_node = Node("Assignment", parent=program)
            Node(f"ID: {val}", parent=assign_node)
            i += 2  # Skip identifier and '='
            expr = Node("Expression", parent=assign_node)
            while i < len(tokens) and tokens[i][1] != 'Delimiter':
                etype, ekind, eval_ = tokens[i]
                if ekind == 'Identifier':
                    Node(f"ID: {eval_}", parent=expr)
                elif ekind == 'Arithmetic_Operator':
                    Node(f"Arithmetic_Operator: {eval_}", parent=expr)
                elif ekind == 'Integer_Constant':
                    Node(f"Integer_Constant: {eval_}", parent=expr)
                elif ekind == 'Float_Constant':
                    Node(f"Float_Constant: {eval_}", parent=expr)
                i += 1
            i += 1  # Skip the ';'
            continue

        # 5) If statement
        if kind == 'Keyword' and val == 'if':
            if_node = Node("if()", parent=program)

            # skip to '('
            i += 1
            while i < n and not (tokens[i][1] == 'Bracket_Parenthesis' and tokens[i][2] == '('):
                i += 1

            # parse condition
            i += 1
            cond = Node("Expression", parent=if_node)
            _, _, lop = tokens[i];
            Node(f"ID: {lop}", parent=cond)
            i += 1
            _, _, rop = tokens[i];
            Node(f"Relational_Operator: {rop}", parent=cond)
            i += 1
            _, _, rip = tokens[i];
            Node(f"Integer_Constant: {rip}", parent=cond)

            # skip to '{'
            i += 1
            while i < n and not (tokens[i][1] == 'Bracket_Parenthesis' and tokens[i][2] == '{'):
                i += 1

            # parse inner assignment
            stmt = Node("Statement", parent=if_node)
            asg = Node("Assignment", parent=stmt)

            # identifier
            i += 1
            _, _, lid = tokens[i];
            Node(f"ID: {lid}", parent=asg)

            # skip '='
            i += 2

            expr2 = Node("Expression", parent=asg)
            _, _, l2 = tokens[i];
            Node(f"ID: {l2}", parent=expr2)
            i += 1
            _, _, a2 = tokens[i];
            Node(f"Arithmetic_Operator: {a2}", parent=expr2)
            i += 1
            _, _, r2 = tokens[i];
            Node(f"Integer_Constant: {r2}", parent=expr2)

            i += 1
            continue

        # 6) Return statement
        if kind == 'Keyword' and val == 'return':
            rtn = Node("return()", parent=program)
            i += 1
            _, _, rv = tokens[i]
            Node(f"Integer_Constant: {rv}", parent=rtn)
            i += 1
            continue

        # 7) End marker
        if kind == 'Keyword' and val == 'End':
            Node("End", parent=program)
            i += 1
            continue

        # default
        i += 1

    return program


# ---------------------------
# Streamlit UI
# ---------------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Statistics", "Parse Tree", "Compare", "Grammar"],
                           icons=["house", "bar-chart", "diagram-3", "columns-gap", "book"],
                           menu_icon="code-slash", default_index=0)

# ---------------------------
# Page: Home
# ---------------------------
if selected == "Home":
    st.title("🧠 C Lexer Analyzer")
    tab1, tab2 = st.tabs(["📁 Upload File", "⌘ Live Typing"])

    def display_output_area(tokens, errors, key_suffix=""):
        st.subheader("🔍 Tokenized Output")
        df = pd.DataFrame(tokens, columns=["Line", "Type", "Value"])
        st.dataframe(df, use_container_width=True)

        st.subheader("🚨 Errors & Warnings")
        if errors:
            err_df = pd.DataFrame(errors, columns=["Line", "Type", "Value"])
            st.dataframe(err_df, use_container_width=True)
        else:
            st.success("No lexical errors found!")

        # Save CSV button with a unique key
        output_path = r"D:\Ammar's Folder\Lexer\tokens.csv"
        if st.button("Save CSV File", key=f"save_csv_{key_suffix}"):
            df.to_csv(output_path, index=False)
            st.success(f"File saved to {output_path}")

    with tab1:
        uploaded_file = st.file_uploader("Upload .c or .cpp file", type=["c", "cpp"])
        if uploaded_file:
            st.session_state.code = uploaded_file.read().decode("utf-8")
            st.code(st.session_state.code, language="c")
            st.session_state.tokens, st.session_state.errors = lex_code(st.session_state.code)
            display_output_area(st.session_state.tokens, st.session_state.errors, key_suffix="upload")

    with tab2:
        live_code = st.text_area("Type your C code:", height=300)
        if live_code:
            st.session_state.code = live_code
            st.session_state.tokens, st.session_state.errors = lex_code(live_code)
            display_output_area(st.session_state.tokens, st.session_state.errors, key_suffix="live")

# ---------------------------
# Page: Statistics
# ---------------------------
elif selected == "Statistics":
    st.title("📊 Token Statistics")
    if st.session_state.tokens:
        stats = get_token_stats(st.session_state.tokens)
        st.write(stats)
        fig = px.pie(names=list(stats.keys()), values=list(stats.values()), title="Token Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload or enter code in the Home tab.")

# ---------------------------
# Page: Parse Tree
# ---------------------------
elif selected == "Parse Tree":
    st.title("🌳 Visual Parse Tree")

    if st.session_state.tokens:
        root = generate_parse_tree(st.session_state.tokens)

        # Convert the tree into a string for display
        tree_output = "\n".join([f"{pre}{node.name}" for pre, _, node in RenderTree(root)])

        # Display the parse tree using Streamlit's st.code()
        st.code(tree_output)

    else:
        st.info("Please upload or enter code in the Home tab.")

# ---------------------------
# Page: Compare
# ---------------------------
elif selected == "Compare":
    st.title("📂 Code Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Code 1")
        file1 = st.file_uploader("Upload .c or .cpp file", type=["c", "cpp"])
        text1 = st.text_area("Or paste Code 1", height=250, key="txt1")

    with col2:
        st.subheader("Code 2")
        file2 = st.file_uploader("Upload .c or .cpp file", type=["c", "cpp"], key="2")
        text2 = st.text_area("Or paste Code 2", height=250, key="txt2")

    code1 = file1.read().decode('utf-8') if file1 else text1
    code2 = file2.read().decode('utf-8') if file2 else text2

    if code1 and code2:
        tokens1, errors1 = lex_code(code1)
        tokens2, errors2 = lex_code(code2)

        df1 = pd.DataFrame(tokens1, columns=["Line", "Type", "Value"])
        df2 = pd.DataFrame(tokens2, columns=["Line", "Type", "Value"])

        st.subheader("🔍 Lexical Token Comparison")

        if df1.equals(df2):
            st.success("✅ Both codes are lexically identical!")
            st.markdown("#### Tokens")
            st.dataframe(df1, use_container_width=True)
        else:
            st.warning("⚠️ The codes are different lexically.")

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### Tokens from Code 1")
                st.dataframe(df1, use_container_width=True)
            with col4:
                st.markdown("#### Tokens from Code 2")
                st.dataframe(df2, use_container_width=True)

            # Differences Table
            st.subheader("🔬 Differences Highlight Table")
            max_len = max(len(tokens1), len(tokens2))
            diff_rows = []
            for i in range(max_len):
                tok1 = tokens1[i] if i < len(tokens1) else ("", "", "")
                tok2 = tokens2[i] if i < len(tokens2) else ("", "", "")
                if tok1 != tok2:
                    diff_rows.append({
                        "Line#": i + 1,
                        "Code 1 Token": f"{tok1[1]}: {tok1[2]}" if tok1[1] else "",
                        "Code 2 Token": f"{tok2[1]}: {tok2[2]}" if tok2[1] else ""
                    })

            if diff_rows:
                diff_df = pd.DataFrame(diff_rows)
                st.dataframe(diff_df, use_container_width=True)
            else:
                st.info("Minor reordering or formatting differences only.")

        # Error section
        st.subheader("🚨 Lexical Errors")
        if df1.equals(df2):
            if errors1:
                st.dataframe(pd.DataFrame(errors1, columns=["Line", "Type", "Value"]))
            else:
                st.success("No lexical errors found in either code.")
        else:
            col5, col6 = st.columns(2)
            with col5:
                if errors1:
                    st.markdown("##### Code 1 Errors")
                    st.dataframe(pd.DataFrame(errors1, columns=["Line", "Type", "Value"]))
                else:
                    st.success("No errors in Code 1.")
            with col6:
                if errors2:
                    st.markdown("##### Code 2 Errors")
                    st.dataframe(pd.DataFrame(errors2, columns=["Line", "Type", "Value"]))
                else:
                    st.success("No errors in Code 2.")

# ---------------------------
# Page: Grammar
# ---------------------------
elif selected == "Grammar":
    st.title("📖 C Grammar Rules")
    st.markdown("""
    ### C Grammar (Informal)

    **1. Declaration**:  
    ```
    type identifier ;
    ```
    **2. Assignment**:  
    ```
    identifier = expression ;
    ```

    **3. Expression**:  
    ```
    term { (+|-) term }*
    ```

    **4. Term**:  
    ```
    factor { (*|/|%) factor }*
    ```

    **5. Factor**:  
    ```
    ( expression ) | constant | identifier
    ```
    **6. Conditional**:  
    ```
    if ( condition ) statement
    ```

    """)