; TypeScript tags. Shared by .ts, .mts, .cts and .tsx.
;
; Everything JavaScript recognises, plus the four declarations that only exist
; in TypeScript — interface, type alias, enum, and abstract class — and the type
; annotations that make its reference graph worth having.

; ---------------------------------------------------------------------------
; Type-level declarations
; ---------------------------------------------------------------------------

(export_statement declaration: (interface_declaration name: (type_identifier) @name)) @definition.type
(interface_declaration name: (type_identifier) @name) @definition.type

(export_statement declaration: (type_alias_declaration name: (type_identifier) @name)) @definition.type
(type_alias_declaration name: (type_identifier) @name) @definition.type

(export_statement declaration: (enum_declaration name: (identifier) @name)) @definition.enum
(enum_declaration name: (identifier) @name) @definition.enum

(interface_body (method_signature name: (property_identifier) @name) @definition.method)
(interface_body (property_signature name: (property_identifier) @name) @definition.field)

; ---------------------------------------------------------------------------
; Classes
; ---------------------------------------------------------------------------

(export_statement declaration: (class_declaration name: (type_identifier) @name)) @definition.class
(export_statement declaration: (abstract_class_declaration name: (type_identifier) @name)) @definition.class
(class_declaration name: (type_identifier) @name) @definition.class
(abstract_class_declaration name: (type_identifier) @name) @definition.class

(class_body (method_definition name: (property_identifier) @name) @definition.method)
(class_body (abstract_method_signature name: (property_identifier) @name) @definition.method)
(class_body (public_field_definition
  name: [(property_identifier) @name (private_property_identifier) @name]) @definition.field)

; ---------------------------------------------------------------------------
; Object literals — the Frappe and Vue idiom, where the real content is not a
; declaration at all.
; ---------------------------------------------------------------------------

(object (method_definition name: (property_identifier) @name) @definition.method)
(object (pair
  key: [(property_identifier) @name (string (string_fragment) @name)]
  value: [(function_expression) (arrow_function)]) @definition.method)

; ---------------------------------------------------------------------------
; Functions
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

(extends_clause value: [(identifier) @name
                        (member_expression property: (property_identifier) @name)]) @reference.class
(implements_clause (type_identifier) @name) @reference.class

(type_annotation [(type_identifier) @name
                  (generic_type name: (type_identifier) @name)]) @reference.type
