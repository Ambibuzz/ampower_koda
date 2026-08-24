; Python tags.
;
; Pattern order is precedence: the earliest pattern that matches a given
; identifier wins. Read top to bottom and the ambiguities resolve themselves —
; a decorated method is a method, not a function that happens to sit in a class.

; ---------------------------------------------------------------------------
; Classes
; ---------------------------------------------------------------------------

; An enumeration is a class, but its members are data a search should surface,
; so it earns its own role. Matched on the trailing base name, which catches
; `enum.IntEnum` and a re-exported `IntEnum` alike.
; The decorated form comes first for the same reason it does everywhere else:
; both patterns bind the same `@name`, the lower pattern index wins, and only
; this one's extent includes `@enum.unique`.
((decorated_definition (class_definition
   name: (identifier) @name
   superclasses: (argument_list
     [(identifier) @_base
      (attribute attribute: (identifier) @_base)]))) @definition.enum
 (#match? @_base "^(Enum|IntEnum|StrEnum|Flag|IntFlag|ReprEnum)$"))

((class_definition
   name: (identifier) @name
   superclasses: (argument_list
     [(identifier) @_base
      (attribute attribute: (identifier) @_base)])) @definition.enum
 (#match? @_base "^(Enum|IntEnum|StrEnum|Flag|IntFlag|ReprEnum)$"))

(decorated_definition (class_definition name: (identifier) @name)) @definition.class
(class_definition name: (identifier) @name) @definition.class

; ---------------------------------------------------------------------------
; Methods — direct children of a class body, so a closure inside a method is
; still a function.
; ---------------------------------------------------------------------------

(class_definition body: (block
  (decorated_definition (function_definition name: (identifier) @name)) @definition.method))

(class_definition body: (block
  (function_definition name: (identifier) @name) @definition.method))

; ---------------------------------------------------------------------------
; Class-level fields. Covers `rate = 1` and `currency: str = "INR"` alike:
; tree-sitter models an annotated assignment as an assignment with a type.
; ---------------------------------------------------------------------------

(class_definition body: (block
  (expression_statement
    (assignment left: (identifier) @name)) @definition.field))

(class_definition body: (block
  (expression_statement
    (assignment left: [(pattern_list (identifier) @name)
                       (tuple_pattern (identifier) @name)])) @definition.field))

; ---------------------------------------------------------------------------
; Module-level type aliases, before constants: `PATH: TypeAlias = str` is a
; type first and an upper-case name second.
; ---------------------------------------------------------------------------

((module (expression_statement
   (assignment
     left: (identifier) @name
     type: (type (identifier) @_alias))) @definition.type)
 (#eq? @_alias "TypeAlias"))

(type_alias_statement left: (type (identifier) @name)) @definition.type

; ---------------------------------------------------------------------------
; Module-level constants. Convention, not analysis — but it is the convention
; every Python codebase already follows, which makes it a better signal than
; any attempt to prove a name is never rebound.
; ---------------------------------------------------------------------------

((module (expression_statement
   (assignment left: (identifier) @name)) @definition.constant)
 (#match? @name "^_*[A-Z][A-Z0-9_]*$"))

((module (expression_statement
   (assignment left: [(pattern_list (identifier) @name)
                      (tuple_pattern (identifier) @name)])) @definition.constant)
 (#match? @name "^_*[A-Z][A-Z0-9_]*$"))

; ---------------------------------------------------------------------------
; Functions
; ---------------------------------------------------------------------------

(decorated_definition (function_definition name: (identifier) @name)) @definition.function
(function_definition name: (identifier) @name) @definition.function

; ---------------------------------------------------------------------------
; References. Three kinds and no more: there is no import edge and no extends
; edge beyond the base-class name itself, because recovering those needs module
; resolution, and a resolver that is wrong occasionally produces a graph that is
; confidently wrong occasionally.
; ---------------------------------------------------------------------------

(call function: (identifier) @name) @reference.call
(call function: (attribute attribute: (identifier) @name)) @reference.call

(class_definition superclasses: (argument_list
  [(identifier) @name
   (attribute attribute: (identifier) @name)])) @reference.class

(type [(identifier) @name
       (attribute attribute: (identifier) @name)
       (generic_type (identifier) @name)]) @reference.type
