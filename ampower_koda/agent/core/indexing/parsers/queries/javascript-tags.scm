; JavaScript tags. Shared by .js, .jsx, .mjs and .cjs.
;
; Frappe client scripts are the shape this is tuned for: a top-level
; `frappe.ui.form.on("DocType", { refresh(frm) {...} })` call whose real content
; is object methods, not declarations. A query that only saw `function` and
; `class` would report almost nothing about a Frappe app's front end.

; ---------------------------------------------------------------------------
; Classes — the exported form first, so `export` lands inside the extent.
; ---------------------------------------------------------------------------

(export_statement declaration: (class_declaration name: (identifier) @name)) @definition.class
(class_declaration name: (identifier) @name) @definition.class

; ---------------------------------------------------------------------------
; Methods: class members, and object literal methods. The second is what a
; Frappe form handler is made of.
; ---------------------------------------------------------------------------

(class_body (method_definition name: (property_identifier) @name) @definition.method)

(class_body (field_definition
  property: [(property_identifier) @name
             (private_property_identifier) @name]) @definition.field)

(object (method_definition name: (property_identifier) @name) @definition.method)

(object (pair
  key: [(property_identifier) @name (string (string_fragment) @name)]
  value: [(function_expression) (arrow_function)]) @definition.method)

; ---------------------------------------------------------------------------
; Functions, including the `const f = () => {}` form that most modern code uses
; in place of a declaration.
; ---------------------------------------------------------------------------

(export_statement declaration: (function_declaration name: (identifier) @name)) @definition.function
(function_declaration name: (identifier) @name) @definition.function
(generator_function_declaration name: (identifier) @name) @definition.function

(export_statement declaration: (lexical_declaration
  (variable_declarator
    name: (identifier) @name
    value: [(function_expression) (arrow_function)]))) @definition.function

(lexical_declaration (variable_declarator
  name: (identifier) @name
  value: [(function_expression) (arrow_function)])) @definition.function

(variable_declaration (variable_declarator
  name: (identifier) @name
  value: [(function_expression) (arrow_function)])) @definition.function

; ---------------------------------------------------------------------------
; Constants. Same convention as Python: SCREAMING_CASE is a deliberate signal a
; developer already writes, and a better one than any analysis of reassignment.
;
; Note the outer parentheses. A predicate must sit *inside* the pattern it
; filters; written one level out it becomes a separate, zero-step, unrooted
; pattern that matches at every node in the file — and the pattern it was meant
; to filter runs unfiltered.
; ---------------------------------------------------------------------------

((export_statement declaration: (lexical_declaration
   (variable_declarator name: (identifier) @name))) @definition.constant
 (#match? @name "^_*[A-Z][A-Z0-9_]*$"))

((lexical_declaration (variable_declarator
   name: (identifier) @name)) @definition.constant
 (#match? @name "^_*[A-Z][A-Z0-9_]*$"))

; ---------------------------------------------------------------------------
; References
; ---------------------------------------------------------------------------

(call_expression function: (identifier) @name) @reference.call
(call_expression function: (member_expression property: (property_identifier) @name)) @reference.call

(new_expression constructor: [(identifier) @name
                              (member_expression property: (property_identifier) @name)]) @reference.class

(class_heritage [(identifier) @name
                 (member_expression property: (property_identifier) @name)]) @reference.class
