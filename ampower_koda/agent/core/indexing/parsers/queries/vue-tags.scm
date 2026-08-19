; Vue single-file component tags.
;
; The Vue grammar parses the template and hands both `<script>` and `<style>`
; back as one node of raw text each. Everything that makes a component worth
; indexing — its name, its methods, its computed properties — lives in that
; text, so the whole of this file is two injections and one convenience.
;
; The JavaScript query then does the real work, including the object-literal
; method patterns that a `export default { methods: { … } }` component is
; entirely made of.

(script_element (raw_text) @injection.javascript)
(style_element (raw_text) @injection.css)
