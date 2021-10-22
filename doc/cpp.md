# C++ code conventions
* Use CamelCase for all names. Start types (such as classes, structs, and typedefs) with a capital letter, other names (functions, variables) with a lowercase letter
* C++ interfaces are named with a Interface suffix, and abstract base classes with an Abstract prefix.
* Member variables are named with a trailing underscore.
* Accessors for a variable foo_ are named foo() and setFoo().
* Global variables are named with a g_ prefix.
* Static class variables are named with a s_ prefix.
* Global constants are often named with a c_ prefix.
* If the main responsibility of a file is to implement a particular class, then the name of the file should match that class, except for possible abbreviations to avoid repetition in file names (e.g., if all classes within a module start with the module name, omitting or abbreviating the module name is OK). Currently, all source file names are lowercase, but this casing difference should be the only difference.

# Common guidelines for C and C++ code
* Preprocessor macros should be all upper-case. Do not use leading underscores, as all such names are reserved according to the C/C++ standard.
* Name include guards like GMX_DIRNAME_HEADERNAME_H.
* Boolean variables are always named with a b prefix, followed by a CamelCase name.
* Enum values are named with an e prefix. For enum types exposed widely in the codebase, this is followed typically by a part that makes the enum values not conflict with other enums in the same scope. In C code, this is typically an all-lowercase acronym (e.g., epbcNONE); in C++, the same approach may be used, or the name of the enum type is used (e.g., eHelpOutputFormat_Console).
* Avoid abbreviations that are not obvious to a general reader.
* If you use acronyms (e.g., PME, DD) in names, follow the Microsoft policy on casing: two letters is uppercase (DD), three or more is lowercase (Pme). If the first letter would be lowercase in the context where it is used (e.g., at the beginning of a function name, or anywhere in a C function name), it is clearest to use all-lowercase acronym.